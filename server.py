from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import time
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'llama2')
EXTERNAL_API_URL = os.getenv('EXTERNAL_API_URL', None)  # URL de API externa si quieres usar una real
EXTERNAL_API_KEY = os.getenv('EXTERNAL_API_KEY', None)

# Modelos disponibles simulados
AVAILABLE_MODELS = [
    {
        "name": "llama2",
        "modified_at": "2024-01-01T00:00:00Z",
        "size": 3826793677,
        "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
        "name": "llama2:7b",
        "modified_at": "2024-01-01T00:00:00Z",
        "size": 3826793677,
        "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
        "name": "codellama",
        "modified_at": "2024-01-01T00:00:00Z",
        "size": 3826793677,
        "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
        "name": "mistral",
        "modified_at": "2024-01-01T00:00:00Z",
        "size": 4109016066,
        "digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }
]

def generate_fake_response(prompt, model="llama2"):
    """
    Genera una respuesta simulada basada en el prompt
    """
    responses = [
        f"Esta es una respuesta simulada del modelo {model} para el prompt: '{prompt[:50]}...'",
        f"Simulando respuesta de {model}: He procesado tu solicitud y aquí está mi respuesta.",
        f"Respuesta generada por {model}: Entiendo tu pregunta y puedo ayudarte con eso.",
        f"Como {model}, puedo decirte que esta es una respuesta de prueba para tu consulta."
    ]
    
    # Seleccionar respuesta basada en hash del prompt para consistencia
    import hashlib
    hash_obj = hashlib.md5(prompt.encode())
    index = int(hash_obj.hexdigest(), 16) % len(responses)
    
    return responses[index]

@app.route('/api/tags', methods=['GET'])
def list_models():
    """
    Endpoint para listar modelos disponibles
    Compatible con: GET /api/tags
    """
    return jsonify({
        "models": AVAILABLE_MODELS
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Endpoint principal para generar texto
    Compatible con: POST /api/generate
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        model = data.get('model', DEFAULT_MODEL)
        prompt = data.get('prompt', '')
        stream = data.get('stream', False)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Verificar si el modelo existe
        model_exists = any(m['name'] == model for m in AVAILABLE_MODELS)
        if not model_exists:
            return jsonify({"error": f"Model '{model}' not found"}), 404
        
        response_text = generate_fake_response(prompt, model)
        
        if stream:
            # Respuesta en streaming
            def generate_stream():
                words = response_text.split()
                for i, word in enumerate(words):
                    chunk = {
                        "model": model,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "response": word + (" " if i < len(words) - 1 else ""),
                        "done": i == len(words) - 1
                    }
                    
                    if chunk["done"]:
                        chunk.update({
                            "total_duration": 1000000000,  # 1 segundo en nanosegundos
                            "load_duration": 100000000,    # 100ms
                            "prompt_eval_count": len(prompt.split()),
                            "prompt_eval_duration": 200000000,  # 200ms
                            "eval_count": len(words),
                            "eval_duration": 800000000      # 800ms
                        })
                    
                    yield json.dumps(chunk) + '\n'
                    time.sleep(0.1)  # Simular latencia
            
            return Response(generate_stream(), mimetype='application/x-ndjson')
        else:
            # Respuesta completa
            return jsonify({
                "model": model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "response": response_text,
                "done": True,
                "total_duration": 1000000000,
                "load_duration": 100000000,
                "prompt_eval_count": len(prompt.split()),
                "prompt_eval_duration": 200000000,
                "eval_count": len(response_text.split()),
                "eval_duration": 800000000
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
        
        if not messages:
            return jsonify({"error": "Messages are required"}), 400
        
        # Extraer el último mensaje del usuario
        last_message = messages[-1]
        prompt = last_message.get('content', '')
        
        response_text = generate_fake_response(prompt, model)
        
        if stream:
            def generate_chat_stream():
                words = response_text.split()
                for i, word in enumerate(words):
                    chunk = {
                        "model": model,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                            "content": word + (" " if i < len(words) - 1 else "")
                        },
                        "done": i == len(words) - 1
                    }
                    
                    if chunk["done"]:
                        chunk.update({
                            "total_duration": 1000000000,
                            "load_duration": 100000000,
                            "prompt_eval_count": len(prompt.split()),
                            "prompt_eval_duration": 200000000,
                            "eval_count": len(words),
                            "eval_duration": 800000000
                        })
                    
                    yield json.dumps(chunk) + '\n'
                    time.sleep(0.1)
            
            return Response(generate_chat_stream(), mimetype='application/x-ndjson')
        else:
            return jsonify({
                "model": model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "done": True,
                "total_duration": 1000000000,
                "load_duration": 100000000,
                "prompt_eval_count": len(prompt.split()),
                "prompt_eval_duration": 200000000,
                "eval_count": len(response_text.split()),
                "eval_duration": 800000000
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
        
        # Buscar el modelo
        model_info = next((m for m in AVAILABLE_MODELS if m['name'] == model_name), None)
        
        if not model_info:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
        
        return jsonify({
            "license": "MIT",
            "modelfile": f"FROM {model_name}",
            "parameters": {
                "num_ctx": 2048,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40
            },
            "template": "{{ .Prompt }}",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pull', methods=['POST'])
def pull_model():
    """
    Endpoint para simular la descarga de un modelo
    Compatible con: POST /api/pull
    """
    try:
        data = request.get_json()
        model_name = data.get('name', DEFAULT_MODEL)
        
        # Simular proceso de descarga
        def simulate_pull():
            statuses = [
                {"status": f"pulling manifest for {model_name}"},
                {"status": f"pulling {model_name}... 10%", "completed": 1000000, "total": 10000000},
                {"status": f"pulling {model_name}... 50%", "completed": 5000000, "total": 10000000},
                {"status": f"pulling {model_name}... 100%", "completed": 10000000, "total": 10000000},
                {"status": f"verifying sha256 digest"},
                {"status": f"writing manifest"},
                {"status": f"removing any unused layers"},
                {"status": "success"}
            ]
            
            for status in statuses:
                yield json.dumps(status) + '\n'
                time.sleep(0.5)
        
        return Response(simulate_pull(), mimetype='application/x-ndjson')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/push', methods=['POST'])
def push_model():
    """
    Endpoint para simular subida de modelo
    Compatible con: POST /api/push
    """
    return jsonify({"error": "Push not implemented in fake service"}), 501

@app.route('/api/embeddings', methods=['POST'])
def embeddings():
    """
    Endpoint para generar embeddings (simulados)
    Compatible con: POST /api/embeddings
    """
    try:
        data = request.get_json()
        model = data.get('model', DEFAULT_MODEL)
        prompt = data.get('prompt', '')
        
        # Generar embeddings falsos (vector de 768 dimensiones)
        import random
        random.seed(hash(prompt))  # Para consistencia
        embedding = [random.uniform(-1, 1) for _ in range(768)]
        
        return jsonify({
            "embedding": embedding
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de health check
    """
    return jsonify({
        "status": "healthy",
        "service": "ollama-fake",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route('/version', methods=['GET'])
@app.route('/api/version', methods=['GET'])
def version():
    """
    Endpoint para obtener la versión
    """
    return jsonify({
        "version": "0.1.0-fake"
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
    
    print(f"🚀 Servicio Ollama Fake iniciando en http://{host}:{port}")
    print(f"📋 Modelos disponibles: {[m['name'] for m in AVAILABLE_MODELS]}")
    print(f"🔧 Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug)
