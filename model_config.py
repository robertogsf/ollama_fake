"""
Configuración de modelos para el servicio Ollama con IA real
"""

# Modelos disponibles con sus configuraciones
REAL_MODELS = {
    "qwen3:14b-gguf": {
        "model_name": "Qwen/Qwen3-14B-GGUF",
        "display_name": "qwen3:14b-gguf",
        "size": 9000000000,  # ~9GB aproximado para Q4_K_M
        "description": "Qwen3 14B GGUF - Modelo optimizado para eficiencia (Q4_K_M)",
        "max_tokens": 4096,
        "temperature": 0.7,
        "torch_dtype": "float16",
        "model_type": "gguf",
        "gguf_filename": "qwen3-14b-gguf-q4_k_m.gguf",
        "use_gpu": True
    },
    "qwen3:14b-instruct": {
        "model_name": "Qwen/Qwen3-14B-Instruct",
        "display_name": "qwen3:14b-instruct",
        "size": 28000000000,  # ~28GB aproximado
        "description": "Qwen3 14B Instruct - Modelo oficial de instrucciones",
        "max_tokens": 4096,
        "temperature": 0.7,
        "torch_dtype": "float16",
        "model_type": "transformers"
    },
    "qwen3:14b": {
        "model_name": "Qwen/Qwen3-14B",
        "display_name": "qwen3:14b",
        "size": 28000000000,  # ~28GB aproximado
        "description": "Qwen3 14B - Modelo conversacional de última generación",
        "max_tokens": 4096,
        "temperature": 0.7,
        "torch_dtype": "float16",
        "model_type": "transformers"
    },
    "qwen2.5:14b": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "display_name": "qwen2.5:14b", 
        "size": 28000000000,  # ~28GB aproximado
        "description": "Qwen2.5 14B Instruct - Modelo conversacional avanzado",
        "max_tokens": 4096,
        "temperature": 0.7,
        "torch_dtype": "float16",
        "model_type": "transformers"
    },
    "qwen2.5:7b": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "display_name": "qwen2.5:7b",
        "size": 14000000000,  # ~14GB aproximado
        "description": "Qwen2.5 7B Instruct - Versión más ligera",
        "max_tokens": 4096,
        "temperature": 0.7,
        "torch_dtype": "float16",
        "model_type": "transformers"
    },
    "qwen2.5:3b": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "display_name": "qwen2.5:3b",
        "size": 6000000000,  # ~6GB aproximado
        "description": "Qwen2.5 3B Instruct - Versión eficiente",
        "max_tokens": 4096,
        "temperature": 0.7,
        "torch_dtype": "float16",
        "model_type": "transformers"
    },
    "llama3.2:3b": {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "display_name": "llama3.2:3b",
        "size": 6000000000,
        "description": "Llama 3.2 3B Instruct",
        "max_tokens": 2048,
        "temperature": 0.7,
        "torch_dtype": "float16",
        "model_type": "transformers"
    }
}

# Modelo por defecto
DEFAULT_MODEL = "qwen2.5:7b"

# Configuración de dispositivo
DEVICE_CONFIG = {
    "auto_device_map": True,
    "low_cpu_mem_usage": True,
    "trust_remote_code": True
}
