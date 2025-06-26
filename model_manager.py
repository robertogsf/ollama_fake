"""
Manager para modelos de IA reales usando Transformers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import gc
import os
from dotenv import load_dotenv
from model_config import REAL_MODELS, DEFAULT_MODEL, DEVICE_CONFIG
from fallback_responses import get_fallback_response

# Cargar variables de entorno
load_dotenv()

# Configurar cache de HuggingFace
if os.getenv('HF_HOME'):
    os.environ['HF_HOME'] = os.getenv('HF_HOME')
if os.getenv('TRANSFORMERS_CACHE'):
    os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE')

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Dispositivo detectado: {self.device}")
        
    def load_model(self, model_key):
        """Cargar un modelo específico"""
        if model_key not in REAL_MODELS:
            raise ValueError(f"Modelo {model_key} no disponible")
            
        model_config = REAL_MODELS[model_key]
        model_name = model_config["model_name"]
        
        # Si ya tenemos este modelo cargado, no hacer nada
        if self.current_model_name == model_key:
            print(f"✅ Modelo {model_key} ya está cargado")
            return True
            
        # Limpiar modelo anterior
        self._unload_current_model()
        
        print(f"📥 Descargando/Cargando modelo: {model_name}")
        print(f"💾 Tamaño estimado: {model_config['size'] / 1e9:.1f}GB")
        
        try:
            # Configurar dtype
            torch_dtype = getattr(torch, model_config.get("torch_dtype", "float16"))
            
            # Cargar tokenizer
            print("🔤 Cargando tokenizer...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=DEVICE_CONFIG["trust_remote_code"]
            )
            
            # Cargar modelo
            print("🧠 Cargando modelo...")
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if DEVICE_CONFIG["auto_device_map"] else None,
                low_cpu_mem_usage=DEVICE_CONFIG["low_cpu_mem_usage"],
                trust_remote_code=DEVICE_CONFIG["trust_remote_code"]
            )
            
            self.current_model_name = model_key
            print(f"✅ Modelo {model_key} cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo {model_key}: {str(e)}")
            self._unload_current_model()
            return False
    
    def _unload_current_model(self):
        """Descargar modelo actual para liberar memoria"""
        if self.current_model is not None:
            print(f"🗑️ Descargando modelo {self.current_model_name}")
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            
            # Limpiar caché de GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def generate_text(self, prompt, model_key=None, max_tokens=None, temperature=None, stream=False):
        """Generar texto usando el modelo cargado"""
        # Usar modelo por defecto si no se especifica
        if model_key is None:
            model_key = DEFAULT_MODEL
            
        # Obtener configuración del modelo
        model_config = REAL_MODELS[model_key]
        max_tokens = max_tokens or model_config.get("max_tokens", 1024)
        temperature = temperature or model_config.get("temperature", 0.7)
        
        # Intentar cargar modelo si es necesario
        model_loaded = self.load_model(model_key)
        
        # Si no se puede cargar el modelo y no es streaming, usar respuesta de respaldo
        if not model_loaded and not stream:
            print(f"⚠️  No se pudo cargar {model_key}, usando respuesta de respaldo")
            return get_fallback_response(model_key, prompt)
        elif not model_loaded and stream:
            print(f"⚠️  No se pudo cargar {model_key}, usando respuesta de respaldo en streaming")
            # Para streaming, devolver la respuesta de respaldo como generador
            def fallback_stream():
                response = get_fallback_response(model_key, prompt)
                words = response.split()
                for word in words:
                    yield word + " "
                    import time
                    time.sleep(0.1)  # Simular latencia
            return fallback_stream()
        
        if self.current_model is None or self.current_tokenizer is None:
            print(f"⚠️  Modelo no disponible, usando respuesta de respaldo")
            if stream:
                def fallback_stream():
                    response = get_fallback_response(model_key, prompt)
                    words = response.split()
                    for word in words:
                        yield word + " "
                        import time
                        time.sleep(0.1)
                return fallback_stream()
            else:
                return get_fallback_response(model_key, prompt)
        
        if stream:
            return self._generate_stream(prompt, model_config, max_tokens, temperature)
        else:
            return self._generate_complete(prompt, model_config, max_tokens, temperature)
    
    def _generate_stream(self, prompt, model_config, max_tokens, temperature):
        """Generar texto en modo streaming"""
        # Formatear prompt para modelos de chat
        if "instruct" in model_config["model_name"].lower():
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = prompt
        
        # Tokenizar
        inputs = self.current_tokenizer.encode(formatted_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        
        # Configurar generación
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
            "pad_token_id": self.current_tokenizer.eos_token_id,
            "eos_token_id": self.current_tokenizer.eos_token_id,
        }
        
        # Generación con streaming
        streamer = TextIteratorStreamer(
            self.current_tokenizer, 
            timeout=60.0, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        generation_kwargs["streamer"] = streamer
        
        # Iniciar generación en hilo separado
        thread = Thread(target=self.current_model.generate, args=(inputs,), kwargs=generation_kwargs)
        thread.start()
        
        # Retornar generador
        for new_text in streamer:
            if new_text:
                yield new_text
    
    def _generate_complete(self, prompt, model_config, max_tokens, temperature):
        """Generar texto completo"""
        try:
            # Formatear prompt para modelos de chat
            if "instruct" in model_config["model_name"].lower():
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = prompt
            
            # Tokenizar
            inputs = self.current_tokenizer.encode(formatted_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to("cuda")
            
            # Configurar generación
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": self.current_tokenizer.eos_token_id,
                "eos_token_id": self.current_tokenizer.eos_token_id,
            }
            
            # Generación completa
            with torch.no_grad():
                outputs = self.current_model.generate(inputs, **generation_kwargs)
            
            # Decodificar solo la parte nueva
            new_tokens = outputs[0][len(inputs[0]):]
            response = self.current_tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"❌ Error en generación real: {e}")
            print("🔄 Usando respuesta de respaldo...")
            # Usar respuesta de respaldo
            return get_fallback_response(self.current_model_name, prompt)
    
    def get_available_models(self):
        """Obtener lista de modelos disponibles"""
        models = []
        for key, config in REAL_MODELS.items():
            models.append({
                "name": config["display_name"],
                "size": config["size"],
                "description": config.get("description", ""),
                "loaded": self.current_model_name == key
            })
        return models
    
    def get_model_info(self, model_key):
        """Obtener información detallada de un modelo"""
        if model_key not in REAL_MODELS:
            return None
            
        config = REAL_MODELS[model_key]
        return {
            "name": config["display_name"],
            "model_name": config["model_name"],
            "size": config["size"],
            "description": config.get("description", ""),
            "max_tokens": config.get("max_tokens", 1024),
            "temperature": config.get("temperature", 0.7),
            "loaded": self.current_model_name == model_key
        }

# Instancia global del manager
model_manager = ModelManager()
