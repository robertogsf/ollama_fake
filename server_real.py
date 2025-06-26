"""
Servidor Ollama con IA Real usando Transformers
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import time
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
from model_manager import model_manager
from model_config import REAL_MODELS, DEFAULT_MODEL

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración
EXTERNAL_API_URL = os.getenv('EXTERNAL_API_URL', None)
EXTERNAL_API_KEY = os.getenv('EXTERNAL_API_KEY', None)

def find_model_key(model_name):
    """
    Encuentra la clave del modelo por nombre o display_name
    """
    # Buscar coincidencia exacta por clave
    if model_name in REAL_MODELS:
        return model_name
    
    # Buscar por display_name
    for key, config in REAL_MODELS.items():
        if config.get('display_name') == model_name:
            return key
    
    return None

def validate_model(model_name):
    """
    Valida que un modelo existe y devuelve su clave
    """
    model_key = find_model_key(model_name)
    if not model_key:
        return None, f"Model '{model_name}' not found. Available models: {list(REAL_MODELS.keys())}"
    return model_key, None

@app.route('/api/tags', methods=['GET'])
def list_models():
    """
    Endpoint para listar modelos disponibles
    Compatible con: GET /api/tags
    """
    try:
        models = []
        for model_key, config in REAL_MODELS.items():
            models.append({
                "name": config["display_name"],
                "modified_at": datetime.utcnow().isoformat() + "Z",
                "size": config["size"],
                "digest": f"sha256:{hash(model_key) % (2**64):016x}",
                "details": {
                    "format": "transformers",
                    "family": "qwen" if "qwen" in model_key else "llama",
                    "parameter_size": config["model_name"].split("-")[-2] if "-" in config["model_name"] else "unknown"
                }
            })
        
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Endpoint principal para generar texto con IA real
    Compatible con: POST /api/generate
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        model = data.get('model', DEFAULT_MODEL)
        prompt = data.get('prompt', '')
        stream = data.get('stream', False)
        max_tokens = data.get('max_tokens', None)
        temperature = data.get('temperature', None)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Verificar si el modelo existe
        model_key, error = validate_model(model)
        if error:
            return jsonify({"error": error}), 404
        
        if stream:
            # Respuesta en streaming
            def generate_stream():
                try:
                    start_time = time.time()
                    full_response = ""
                    
                    for chunk in model_manager.generate_text(
                        prompt=prompt, 
                        model_key=model_key, 
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
                    ):
                        full_response += chunk
                        response_chunk = {
                            "model": model,
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "response": chunk,
                            "done": False
                        }
                        yield json.dumps(response_chunk) + '\n'
                    
                    # Chunk final
                    end_time = time.time()
                    duration_ns = int((end_time - start_time) * 1e9)
                    
                    final_chunk = {
                        "model": model,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "response": "",
                        "done": True,
                        "total_duration": duration_ns,
                        "load_duration": duration_ns // 10,
                        "prompt_eval_count": len(prompt.split()),
                        "prompt_eval_duration": duration_ns // 5,
                        "eval_count": len(full_response.split()),
                        "eval_duration": duration_ns * 4 // 5
                    }
                    yield json.dumps(final_chunk) + '\n'
                    
                except Exception as e:
                    error_chunk = {
                        "model": model,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "response": "",
                        "done": True,
                        "error": str(e)
                    }
                    yield json.dumps(error_chunk) + '\n'
            
            return Response(generate_stream(), mimetype='application/x-ndjson')
        else:
            # Respuesta completa
            start_time = time.time()
            response_text = model_manager.generate_text(
                prompt=prompt,
                model_key=model_key,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            end_time = time.time()
            duration_ns = int((end_time - start_time) * 1e9)
            
            return jsonify({
                "model": model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "response": response_text,
                "done": True,
                "total_duration": duration_ns,
                "load_duration": duration_ns // 10,
                "prompt_eval_count": len(prompt.split()),
                "prompt_eval_duration": duration_ns // 5,
                "eval_count": len(response_text.split()),
                "eval_duration": duration_ns * 4 // 5
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint para chat con formato de mensajes
    Compatible con: POST /api/chat
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        model = data.get('model', DEFAULT_MODEL)
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        max_tokens = data.get('max_tokens', None)
        temperature = data.get('temperature', None)
        
        if not messages:
            return jsonify({"error": "Messages are required"}), 400
        
        # Verificar si el modelo existe
        model_key, error = validate_model(model)
        if error:
            return jsonify({"error": error}), 404
        
        # Convertir mensajes a prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prompt_parts.append(f"Usuario: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Asistente: {content}")
            elif role == 'system':
                prompt_parts.append(f"Sistema: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAsistente:"
        
        if stream:
            def generate_chat_stream():
                try:
                    start_time = time.time()
                    full_response = ""
                    
                    for chunk in model_manager.generate_text(
                        prompt=prompt,
                        model_key=model_key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
                    ):
                        full_response += chunk
                        response_chunk = {
                            "model": model,
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "message": {
                                "role": "assistant",
                                "content": chunk
                            },
                            "done": False
                        }
                        yield json.dumps(response_chunk) + '\n'
                    
                    # Chunk final
                    end_time = time.time()
                    duration_ns = int((end_time - start_time) * 1e9)
                    
                    final_chunk = {
                        "model": model,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "done": True,
                        "total_duration": duration_ns,
                        "load_duration": duration_ns // 10,
                        "prompt_eval_count": len(prompt.split()),
                        "prompt_eval_duration": duration_ns // 5,
                        "eval_count": len(full_response.split()),
                        "eval_duration": duration_ns * 4 // 5
                    }
                    yield json.dumps(final_chunk) + '\n'
                    
                except Exception as e:
                    error_chunk = {
                        "model": model,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "done": True,
                        "error": str(e)
                    }
                    yield json.dumps(error_chunk) + '\n'
            
            return Response(generate_chat_stream(), mimetype='application/x-ndjson')
        else:
            start_time = time.time()
            response_text = model_manager.generate_text(
                prompt=prompt,
                model_key=model_key,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            end_time = time.time()
            duration_ns = int((end_time - start_time) * 1e9)
            
            return jsonify({
                "model": model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "done": True,
                "total_duration": duration_ns,
                "load_duration": duration_ns // 10,
                "prompt_eval_count": len(prompt.split()),
                "prompt_eval_duration": duration_ns // 5,
                "eval_count": len(response_text.split()),
                "eval_duration": duration_ns * 4 // 5
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/show', methods=['POST'])
def show_model():
    """
    Endpoint para mostrar información de un modelo
    Compatible con: POST /api/show
    """
    try:
        data = request.get_json()
        model_name = data.get('name', DEFAULT_MODEL)
        
        model_info = model_manager.get_model_info(model_name)
        
        if not model_info:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
        
        return jsonify({
            "license": "Apache 2.0",
            "modelfile": f"FROM {model_info['model_name']}",
            "parameters": {
                "num_ctx": model_info["max_tokens"],
                "temperature": model_info["temperature"],
                "top_p": 0.9,
                "top_k": 40
            },
            "template": "{{ .Prompt }}",
            "details": {
                "format": "transformers",
                "family": "qwen" if "qwen" in model_name else "llama",
                "parameter_size": model_info['model_name'].split('-')[-2] if '-' in model_info['model_name'] else "unknown",
                "quantization_level": "float16"
            },
            "model_info": model_info
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pull', methods=['POST'])
def pull_model():
    """
    Endpoint para descargar/cargar un modelo
    Compatible con: POST /api/pull
    """
    try:
        data = request.get_json()
        model_name = data.get('name', DEFAULT_MODEL)
        
        if model_name not in REAL_MODELS:
            return jsonify({"error": f"Model '{model_name}' not available"}), 404
        
        def simulate_pull():
            yield json.dumps({"status": f"pulling manifest for {model_name}"}) + '\n'
            yield json.dumps({"status": f"downloading model {model_name}..."}) + '\n'
            
            # Intentar cargar el modelo
            try:
                success = model_manager.load_model(model_name)
                if success:
                    yield json.dumps({"status": "verifying model"}) + '\n'
                    yield json.dumps({"status": "model loaded successfully"}) + '\n'
                    yield json.dumps({"status": "success"}) + '\n'
                else:
                    yield json.dumps({"status": "error", "error": "Failed to load model"}) + '\n'
            except Exception as e:
                yield json.dumps({"status": "error", "error": str(e)}) + '\n'
        
        return Response(simulate_pull(), mimetype='application/x-ndjson')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/embeddings', methods=['POST'])
def embeddings():
    """
    Endpoint para generar embeddings (no implementado para modelos generativos)
    Compatible con: POST /api/embeddings
    """
    return jsonify({"error": "Embeddings not supported by generative models"}), 501

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de health check
    """
    gpu_available = "Si" if model_manager.device == "cuda" else "No"
    current_model = model_manager.current_model_name or "Ninguno"
    
    return jsonify({
        "status": "healthy",
        "service": "ollama-real-ai",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "gpu_available": gpu_available,
        "device": model_manager.device,
        "current_model": current_model
    })

@app.route('/version', methods=['GET'])
@app.route('/api/version', methods=['GET'])
def version():
    """
    Endpoint para obtener la versión
    """
    return jsonify({
        "version": "1.0.0-real-ai"
    })

@app.route('/api/models/unload', methods=['POST'])
def unload_model():
    """
    Endpoint personalizado para descargar modelo de memoria
    """
    try:
        model_manager._unload_current_model()
        return jsonify({"status": "Model unloaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """
    Endpoint raíz - Compatible con verificaciones de Ollama
    """
    return jsonify({
        "message": "Ollama is running",
        "version": "1.0.0-real-ai",
        "status": "healthy"
    })

@app.route('/api', methods=['GET'])
def api_root():
    """
    Endpoint /api - información de la API
    """
    return jsonify({
        "message": "Ollama API",
        "version": "1.0.0-real-ai",
        "endpoints": [
            "/api/tags",
            "/api/generate", 
            "/api/chat",
            "/api/show",
            "/api/pull",
            "/api/embeddings"
        ]
    })

@app.route('/api/ps', methods=['GET'])
def running_models():
    """
    Endpoint para listar modelos en ejecución
    Compatible con: GET /api/ps
    """
    try:
        current_model = model_manager.current_model_name or None
        models = []
        
        if current_model:
            model_config = REAL_MODELS.get(current_model)
            if model_config:
                models.append({
                    "name": model_config["display_name"],
                    "model": model_config["model_name"],
                    "size": model_config["size"],
                    "digest": f"sha256:{hash(current_model) % (2**64):016x}",
                    "details": {
                        "format": "transformers",
                        "family": "qwen" if "qwen" in current_model else "llama",
                        "parameter_size": model_config["model_name"].split("-")[-2] if "-" in model_config["model_name"] else "unknown"
                    },
                    "expires_at": "0001-01-01T00:00:00Z"
                })
        
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """
    Endpoint de estado general
    """
    return jsonify({
        "status": "running",
        "models_loaded": 1 if model_manager.current_model_name else 0,
        "current_model": model_manager.current_model_name or "none",
        "device": model_manager.device,
        "gpu_available": model_manager.device == "cuda"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 11434))  # Puerto por defecto de Ollama
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print("🚀 Servicio Ollama con IA Real iniciando...")
    print(f"🌐 Servidor: http://{host}:{port}")
    print(f"🤖 Modelos disponibles: {list(REAL_MODELS.keys())}")
    print(f"🔧 Dispositivo: {model_manager.device}")
    print(f"🐛 Debug mode: {debug}")
    print("⚠️  NOTA: El primer uso de un modelo tomará tiempo en descargar")
    
    app.run(host=host, port=port, debug=debug)
