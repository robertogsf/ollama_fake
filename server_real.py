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
import logging
from dotenv import load_dotenv
from model_manager import model_manager
from model_config import REAL_MODELS, DEFAULT_MODEL

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    print(f"🎯 ENDPOINT: GET /api/tags - Listando modelos disponibles")
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
        
        response = {"models": models}
        print(f"✅ RESPUESTA /api/tags: {len(models)} modelos encontrados")
        return jsonify(response)
    except Exception as e:
        error_response = {"error": str(e)}
        print(f"❌ ERROR /api/tags: {str(e)}")
        return jsonify(error_response), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Endpoint principal para generar texto con IA real
    Compatible con: POST /api/generate
    """
    print(f"🎯 ENDPOINT: POST /api/generate - Generando texto con IA")
    try:
        data = request.get_json()
        print(f"📝 Datos recibidos: modelo={data.get('model', 'N/A')}, prompt_length={len(data.get('prompt', ''))}, stream={data.get('stream', False)}")
        
        if not data:
            print("❌ ERROR /api/generate: No se proporcionaron datos")
            return jsonify({"error": "No data provided"}), 400
        
        model = data.get('model', DEFAULT_MODEL)
        prompt = data.get('prompt', '')
        stream = data.get('stream', False)
        max_tokens = data.get('max_tokens', None)
        temperature = data.get('temperature', None)
        
        if not prompt:
            print("❌ ERROR /api/generate: Prompt requerido")
            return jsonify({"error": "Prompt is required"}), 400
        
        # Verificar si el modelo existe
        model_key, error = validate_model(model)
        if error:
            print(f"❌ ERROR /api/generate: {error}")
            return jsonify({"error": error}), 404
        
        print(f"🤖 Usando modelo: {model_key}")
        
        if stream:
            print("🌊 Modo streaming activado")
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
                    print(f"✅ Stream completado en {end_time - start_time:.2f}s - Tokens generados: {len(full_response)}")
                    
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
                    print(f"❌ ERROR en stream: {str(e)}")
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
            print("📄 Modo respuesta completa")
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
            
            response = {
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
            }
            
            print(f"✅ RESPUESTA /api/generate: {len(response_text)} caracteres generados en {end_time - start_time:.2f}s")
            return jsonify(response)
    
    except Exception as e:
        print(f"❌ ERROR /api/generate: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint para chat con formato de mensajes
    Compatible con: POST /api/chat
    """
    print(f"🎯 ENDPOINT: POST /api/chat - Chat con formato de mensajes")
    try:
        data = request.get_json()
        print(f"💬 Datos recibidos: modelo={data.get('model', 'N/A')}, num_mensajes={len(data.get('messages', []))}, stream={data.get('stream', False)}")
        
        if not data:
            print("❌ ERROR /api/chat: No se proporcionaron datos")
            return jsonify({"error": "No data provided"}), 400
        
        model = data.get('model', DEFAULT_MODEL)
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        max_tokens = data.get('max_tokens', None)
        temperature = data.get('temperature', None)
        
        if not messages:
            print("❌ ERROR /api/chat: Mensajes requeridos")
            return jsonify({"error": "Messages are required"}), 400
        
        # Verificar si el modelo existe
        model_key, error = validate_model(model)
        if error:
            print(f"❌ ERROR /api/chat: {error}")
            return jsonify({"error": error}), 404
        
        print(f"🤖 Usando modelo: {model_key} para chat")
        
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
        print(f"📝 Prompt generado: {len(prompt)} caracteres")
        
        if stream:
            print("🌊 Modo chat streaming activado")
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
                    print(f"✅ Chat stream completado en {end_time - start_time:.2f}s")
                    
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
                    print(f"❌ ERROR en chat stream: {str(e)}")
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
            print("📄 Modo chat respuesta completa")
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
            
            response = {
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
            }
            
            print(f"✅ RESPUESTA /api/chat: {len(response_text)} caracteres generados en {end_time - start_time:.2f}s")
            return jsonify(response)
    
    except Exception as e:
        print(f"❌ ERROR /api/chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/show', methods=['POST'])
def show_model():
    """
    Endpoint para mostrar información de un modelo
    Compatible con: POST /api/show
    """
    print(f"🎯 ENDPOINT: POST /api/show - Mostrando información del modelo")
    try:
        data = request.get_json()
        model_name = data.get('name', DEFAULT_MODEL)
        print(f"📋 Consultando información del modelo: {model_name}")
        
        model_info = model_manager.get_model_info(model_name)
        
        if not model_info:
            print(f"❌ ERROR /api/show: Modelo '{model_name}' no encontrado")
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
        
        response = {
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
        }
        
        print(f"✅ RESPUESTA /api/show: Información del modelo {model_name} obtenida")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ ERROR /api/show: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pull', methods=['POST'])
def pull_model():
    """
    Endpoint para descargar/cargar un modelo
    Compatible con: POST /api/pull
    """
    print(f"🎯 ENDPOINT: POST /api/pull - Descargando/cargando modelo")
    try:
        data = request.get_json()
        model_name = data.get('name', DEFAULT_MODEL)
        print(f"📥 Intentando cargar modelo: {model_name}")
        
        if model_name not in REAL_MODELS:
            print(f"❌ ERROR /api/pull: Modelo '{model_name}' no disponible")
            return jsonify({"error": f"Model '{model_name}' not available"}), 404
        
        def simulate_pull():
            print(f"🔄 Iniciando descarga simulada del modelo {model_name}")
            yield json.dumps({"status": f"pulling manifest for {model_name}"}) + '\n'
            yield json.dumps({"status": f"downloading model {model_name}..."}) + '\n'
            
            # Intentar cargar el modelo
            try:
                success = model_manager.load_model(model_name)
                if success:
                    print(f"✅ Modelo {model_name} cargado exitosamente")
                    yield json.dumps({"status": "verifying model"}) + '\n'
                    yield json.dumps({"status": "model loaded successfully"}) + '\n'
                    yield json.dumps({"status": "success"}) + '\n'
                else:
                    print(f"❌ Falló la carga del modelo {model_name}")
                    yield json.dumps({"status": "error", "error": "Failed to load model"}) + '\n'
            except Exception as e:
                print(f"❌ ERROR al cargar modelo {model_name}: {str(e)}")
                yield json.dumps({"status": "error", "error": str(e)}) + '\n'
        
        return Response(simulate_pull(), mimetype='application/x-ndjson')
    
    except Exception as e:
        print(f"❌ ERROR /api/pull: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/embeddings', methods=['POST'])
def embeddings():
    """
    Endpoint para generar embeddings (no implementado para modelos generativos)
    Compatible con: POST /api/embeddings
    """
    print(f"🎯 ENDPOINT: POST /api/embeddings - Embeddings no soportados")
    print("⚠️  RESPUESTA /api/embeddings: Embeddings no soportados por modelos generativos")
    return jsonify({"error": "Embeddings not supported by generative models"}), 501

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de health check
    """
    print(f"🎯 ENDPOINT: GET /health - Health check")
    gpu_available = "Si" if model_manager.device == "cuda" else "No"
    current_model = model_manager.current_model_name or "Ninguno"
    
    response = {
        "status": "healthy",
        "service": "ollama-real-ai",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "gpu_available": gpu_available,
        "device": model_manager.device,
        "current_model": current_model
    }
    
    print(f"✅ RESPUESTA /health: Servicio saludable - GPU: {gpu_available}, Modelo actual: {current_model}")
    return jsonify(response)

@app.route('/version', methods=['GET'])
@app.route('/api/version', methods=['GET'])
def version():
    """
    Endpoint para obtener la versión
    """
    print(f"🎯 ENDPOINT: GET /version o /api/version - Consultando versión")
    response = {"version": "1.0.0-real-ai"}
    print(f"✅ RESPUESTA /version: {response}")
    return jsonify(response)

@app.route('/api/models/unload', methods=['POST'])
def unload_model():
    """
    Endpoint personalizado para descargar modelo de memoria
    """
    print(f"🎯 ENDPOINT: POST /api/models/unload - Descargando modelo de memoria")
    try:
        model_manager._unload_current_model()
        response = {"status": "Model unloaded successfully"}
        print(f"✅ RESPUESTA /api/models/unload: Modelo descargado exitosamente")
        return jsonify(response)
    except Exception as e:
        print(f"❌ ERROR /api/models/unload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """
    Endpoint raíz - Compatible con verificaciones de Ollama
    """
    print(f"🎯 ENDPOINT: GET / - Endpoint raíz")
    response = {
        "message": "Ollama is running",
        "version": "1.0.0-real-ai",
        "status": "healthy"
    }
    print(f"✅ RESPUESTA /: Ollama ejecutándose correctamente")
    return jsonify(response)

@app.route('/api', methods=['GET'])
def api_root():
    """
    Endpoint /api - información de la API
    """
    print(f"🎯 ENDPOINT: GET /api - Información de la API")
    response = {
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
    }
    print(f"✅ RESPUESTA /api: Información de la API disponible")
    return jsonify(response)

@app.route('/api/ps', methods=['GET'])
def running_models():
    """
    Endpoint para listar modelos en ejecución
    Compatible con: GET /api/ps
    """
    print(f"🎯 ENDPOINT: GET /api/ps - Listando modelos en ejecución")
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
        
        response = {"models": models}
        print(f"✅ RESPUESTA /api/ps: {len(models)} modelos en ejecución - Modelo actual: {current_model or 'Ninguno'}")
        return jsonify(response)
    except Exception as e:
        print(f"❌ ERROR /api/ps: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """
    Endpoint de estado general
    """
    print(f"🎯 ENDPOINT: GET /api/status - Estado general del servicio")
    response = {
        "status": "running",
        "models_loaded": 1 if model_manager.current_model_name else 0,
        "current_model": model_manager.current_model_name or "none",
        "device": model_manager.device,
        "gpu_available": model_manager.device == "cuda"
    }
    print(f"✅ RESPUESTA /api/status: Estado={response['status']}, Modelos cargados={response['models_loaded']}, Dispositivo={response['device']}")
    return jsonify(response)

@app.errorhandler(404)
def not_found(error):
    print(f"❌ ERROR 404: Endpoint no encontrado - {request.method} {request.path}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"❌ ERROR 500: Error interno del servidor - {request.method} {request.path}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 11434))  # Puerto por defecto de Ollama
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("🚀 Servicio Ollama con IA Real iniciando...")
    print(f"🌐 Servidor: http://{host}:{port}")
    print(f"🤖 Modelos disponibles: {list(REAL_MODELS.keys())}")
    print(f"🔧 Dispositivo: {model_manager.device}")
    print(f"🐛 Debug mode: {debug}")
    print("⚠️  NOTA: El primer uso de un modelo tomará tiempo en descargar")
    print("📝 LOGS DE ENDPOINTS ACTIVADOS - Verás cada llamada y respuesta")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=debug)
