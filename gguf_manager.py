"""
Manager para model            # Configuración para modelo GGUF con GPU habilitada
            gguf_config = {
                "model_path": model_path,
                "n_ctx": model_config.get("max_tokens", 4096),
                "n_threads": 8,
                "n_gpu_layers": -1,  # -1 = usar TODAS las capas en GPU
                "verbose": False,
                "use_mlock": False,
                "use_mmap": True,
                "n_batch": 512,
                "seed": -1,
            }
            
            # Crear instancia de Llama con GPU habilitada
            print(f"🚀 Cargando modelo GGUF con GPU (todas las capas)...")
            model = Llama(**gguf_config)lama-cpp-python
"""
import os
from llama_cpp import Llama
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GGUFManager:
    def __init__(self):
        self.models: Dict[str, Llama] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
    
    def load_model(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        """
        Carga un modelo GGUF
        """
        try:
            logger.info(f"🔄 Cargando modelo GGUF: {model_path}")
            
            # Configuración para modelo GGUF con GPU habilitada
            gguf_config = {
                "model_path": model_path,
                "n_ctx": 4096,  # Contexto completo
                "n_threads": 8,  # Más threads
                "n_gpu_layers": 35,  # Usar GPU para la mayoría de layers (conservador)
                "verbose": False,  # Menos verbose ahora que funciona
                "use_mlock": False,  # Desactivar mlock
                "use_mmap": True,
                "n_batch": 512,  # Batch más grande
                "seed": -1,  # Seed aleatorio
            }
            
            # Crear instancia de Llama en modo CPU para evitar segfaults
            print(f"� Cargando modelo GGUF en modo CPU...")
            model = Llama(**gguf_config)
            
            # Guardar el modelo
            self.models[model_path] = model
            self.model_configs[model_path] = model_config
            
            logger.info(f"✅ Modelo GGUF cargado exitosamente: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo GGUF {model_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_response(self, model_path: str, prompt: str, **kwargs) -> str:
        """
        Genera una respuesta usando un modelo GGUF
        """
        if model_path not in self.models:
            raise ValueError(f"Modelo {model_path} no está cargado")
        
        model = self.models[model_path]
        config = self.model_configs[model_path]
        
        # Parámetros de generación
        generation_params = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", config.get("max_tokens", 512)),
            "temperature": kwargs.get("temperature", config.get("temperature", 0.7)),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", ["</s>", "<|im_end|>"]),
            "echo": False,
        }
        
        try:
            response = model(**generation_params)
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"❌ Error generando respuesta: {str(e)}")
            raise
    
    def stream_response(self, model_path: str, prompt: str, **kwargs):
        """
        Genera una respuesta en streaming usando un modelo GGUF
        """
        if model_path not in self.models:
            raise ValueError(f"Modelo {model_path} no está cargado")
        
        model = self.models[model_path]
        config = self.model_configs[model_path]
        
        # Parámetros de generación
        generation_params = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", config.get("max_tokens", 512)),
            "temperature": kwargs.get("temperature", config.get("temperature", 0.7)),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", ["</s>", "<|im_end|>"]),
            "stream": True,
            "echo": False,
        }
        
        try:
            for token in model(**generation_params):
                yield token["choices"][0]["text"]
        except Exception as e:
            logger.error(f"❌ Error en streaming: {str(e)}")
            raise
    
    def unload_model(self, model_path: str):
        """
        Descarga un modelo de la memoria
        """
        if model_path in self.models:
            del self.models[model_path]
            del self.model_configs[model_path]
            logger.info(f"🗑️ Modelo GGUF descargado: {model_path}")
    
    def is_model_loaded(self, model_path: str) -> bool:
        """
        Verifica si un modelo está cargado
        """
        return model_path in self.models
    
    def get_loaded_models(self) -> list:
        """
        Obtiene la lista de modelos cargados
        """
        return list(self.models.keys())
