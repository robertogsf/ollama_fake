"""
Manager para modelos de IA reales usando Transformers y GGUF
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import gc
import os
from dotenv import load_dotenv
from model_config import REAL_MODELS, DEFAULT_MODEL, DEVICE_CONFIG
from fallback_responses import get_fallback_response
from gguf_manager import GGUFManager
import requests

# Cargar variables de entorno
load_dotenv()

# Configurar cache de HuggingFace
if os.getenv('HF_HOME'):
    os.environ['HF_HOME'] = os.getenv('HF_HOME')
if os.getenv('TRANSFORMERS_CACHE'):
    os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE')

class ModelManager:
    def __init__(self):
        # Modelos Transformers
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
        # Manager GGUF
        self.gguf_manager = GGUFManager()
        self._current_gguf_path = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Dispositivo detectado: {self.device}")
        
    def _download_gguf_file(self, repo_name, filename, local_path):
        """Descargar archivo GGUF desde Hugging Face"""
        models_dir = os.path.dirname(local_path)
        
        # Verificar si ya existe algún archivo GGUF en el directorio
        if os.path.exists(models_dir):
            existing_files = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
            if existing_files:
                existing_path = os.path.join(models_dir, existing_files[0])
                print(f"✅ Archivo GGUF ya existe: {existing_path}")
                return existing_path  # Retornar la ruta del archivo existente
        
        if os.path.exists(local_path):
            print(f"✅ Archivo GGUF ya existe: {local_path}")
            return local_path
        
        try:
            print(f"📥 Descargando archivo GGUF: {filename}")
            
            # Primero, intentar obtener la lista de archivos disponibles
            api_url = f"https://huggingface.co/api/models/{repo_name}"
            actual_filename = filename
            try:
                api_response = requests.get(api_url)
                if api_response.status_code == 200:
                    model_info = api_response.json()
                    if 'siblings' in model_info:
                        gguf_files = [f['rfilename'] for f in model_info['siblings'] if f['rfilename'].endswith('.gguf')]
                        if gguf_files:
                            print(f"📋 Archivos GGUF disponibles: {gguf_files}")
                            # Si el archivo solicitado no existe, usar el primero disponible
                            if filename not in gguf_files:
                                print(f"⚠️  Archivo {filename} no encontrado, usando {gguf_files[0]}")
                                actual_filename = gguf_files[0]
                                # Actualizar el path local
                                local_path = os.path.join(models_dir, actual_filename)
            except Exception as e:
                print(f"⚠️  No se pudo obtener lista de archivos: {e}")
            
            url = f"https://huggingface.co/{repo_name}/resolve/main/{actual_filename}"
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            os.makedirs(models_dir, exist_ok=True)
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r📥 Descargando: {progress:.1f}%", end="", flush=True)
            
            print(f"\n✅ Descarga completada: {local_path}")
            return local_path  # Retornar la ruta del archivo descargado
            
        except Exception as e:
            print(f"❌ Error descargando GGUF: {str(e)}")
            return None
        
    def load_model(self, model_key):
        """Cargar un modelo específico (Transformers o GGUF)"""
        if model_key not in REAL_MODELS:
            raise ValueError(f"Modelo {model_key} no disponible")
            
        model_config = REAL_MODELS[model_key]
        model_name = model_config["model_name"]
        model_type = model_config.get("model_type", "transformers")
        
        # Si ya tenemos este modelo cargado, no hacer nada
        if self.current_model_name == model_key:
            print(f"✅ Modelo {model_key} ya está cargado")
            return True
            
        # Limpiar modelo anterior
        self._unload_current_model()
        
        print(f"📥 Descargando/Cargando modelo: {model_name}")
        print(f"💾 Tamaño estimado: {model_config['size'] / 1e9:.1f}GB")
        print(f"🔧 Tipo de modelo: {model_type}")
        
        if model_type == "gguf":
            return self._load_gguf_model(model_key, model_config)
        else:
            return self._load_transformers_model(model_key, model_config)
    
    def _load_gguf_model(self, model_key, model_config):
        """Cargar modelo GGUF"""
        try:
            model_name = model_config["model_name"]
            gguf_filename = model_config.get("gguf_filename", "model.gguf")
            
            # Crear directorio local para el modelo
            models_dir = "./models_cache/gguf"
            os.makedirs(models_dir, exist_ok=True)
            
            # Path local del archivo GGUF
            local_gguf_path = os.path.join(models_dir, f"{model_key}_{gguf_filename}")
            
            # Descargar archivo GGUF si no existe
            actual_path = self._download_gguf_file(model_name, gguf_filename, local_gguf_path)
            if not actual_path:
                return False
            
            # Guardar la ruta real para usar en generación
            self._current_gguf_path = actual_path
            
            # Cargar modelo con GGUF manager
            if self.gguf_manager.load_model(actual_path, model_config):
                self.current_model_name = model_key
                print(f"✅ Modelo GGUF {model_key} cargado exitosamente")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error cargando modelo GGUF {model_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_transformers_model(self, model_key, model_config):
        """Cargar modelo Transformers"""
        try:
            model_name = model_config["model_name"]
            
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
            print(f"✅ Modelo Transformers {model_key} cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo Transformers {model_key}: {str(e)}")
            return False
    
    def _unload_current_model(self):
        """Descargar modelo actual para liberar memoria"""
        if self.current_model_name is not None:
            print(f"🗑️ Descargando modelo {self.current_model_name}")
            
            # Verificar tipo de modelo y descargar apropiadamente
            if self.current_model_name in REAL_MODELS:
                model_config = REAL_MODELS[self.current_model_name]
                model_type = model_config.get("model_type", "transformers")
                
                if model_type == "gguf":
                    # Descargar modelo GGUF usando la ruta real
                    if self._current_gguf_path:
                        self.gguf_manager.unload_model(self._current_gguf_path)
                        self._current_gguf_path = None
                else:
                    # Descargar modelo Transformers
                    if self.current_model is not None:
                        del self.current_model
                    if self.current_tokenizer is not None:
                        del self.current_tokenizer
            
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            
            # Limpiar caché de GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def generate_text(self, prompt, model_key=None, max_tokens=None, temperature=None, stream=False):
        """Generar texto usando el modelo cargado (Transformers o GGUF)"""
        # Usar modelo por defecto si no se especifica
        if model_key is None:
            model_key = DEFAULT_MODEL
            
        # Obtener configuración del modelo
        model_config = REAL_MODELS[model_key]
        max_tokens = max_tokens or model_config.get("max_tokens", 1024)
        temperature = temperature or model_config.get("temperature", 0.7)
        model_type = model_config.get("model_type", "transformers")
        
        # Intentar cargar modelo si es necesario
        model_loaded = self.load_model(model_key)
        
        # Si no se puede cargar el modelo, usar respuesta de respaldo
        if not model_loaded:
            print(f"⚠️  No se pudo cargar {model_key}, usando respuesta de respaldo")
            if stream:
                def fallback_stream():
                    response = get_fallback_response(model_key, prompt)
                    words = response.split()
                    for word in words:
                        yield word + " "
                        import time
                        time.sleep(0.1)  # Simular latencia
                return fallback_stream()
            else:
                return get_fallback_response(model_key, prompt)
        
        # Generar según el tipo de modelo
        if model_type == "gguf":
            return self._generate_gguf(prompt, model_config, max_tokens, temperature, stream)
        else:
            return self._generate_transformers(prompt, model_config, max_tokens, temperature, stream)
    
    def _generate_gguf(self, prompt, model_config, max_tokens, temperature, stream):
        """Generar texto usando modelo GGUF"""
        try:
            # Usar la ruta real guardada durante la carga
            local_gguf_path = self._current_gguf_path
            if not local_gguf_path:
                # Fallback: construir la ruta esperada
                models_dir = "./models_cache/gguf"
                gguf_filename = model_config.get("gguf_filename", "model.gguf")
                local_gguf_path = os.path.join(models_dir, f"{self.current_model_name}_{gguf_filename}")
            
            if stream:
                return self.gguf_manager.stream_response(
                    local_gguf_path, 
                    prompt, 
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
            else:
                return self.gguf_manager.generate_response(
                    local_gguf_path, 
                    prompt, 
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
        except Exception as e:
            print(f"❌ Error generando con GGUF: {str(e)}")
            return get_fallback_response(self.current_model_name, prompt)
    
    def _generate_transformers(self, prompt, model_config, max_tokens, temperature, stream):
        """Generar texto usando modelo Transformers"""
        if self.current_model is None or self.current_tokenizer is None:
            print(f"⚠️  Modelo Transformers no disponible, usando respuesta de respaldo")
            if stream:
                def fallback_stream():
                    response = get_fallback_response(self.current_model_name, prompt)
                    words = response.split()
                    for word in words:
                        yield word + " "
                        import time
                        time.sleep(0.1)
                return fallback_stream()
            else:
                return get_fallback_response(self.current_model_name, prompt)
        
        if stream:
            return self._generate_transformers_stream(prompt, model_config, max_tokens, temperature)
        else:
            return self._generate_transformers_complete(prompt, model_config, max_tokens, temperature)

    def _generate_transformers_stream(self, prompt, model_config, max_tokens, temperature):
        """Generar texto en modo streaming con Transformers"""
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
    
    def _generate_transformers_complete(self, prompt, model_config, max_tokens, temperature):
        """Generar texto completo con Transformers"""
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
            print(f"❌ Error en generación Transformers: {e}")
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
            "model_type": config.get("model_type", "transformers"),
            "loaded": self.current_model_name == model_key
        }

# Instancia global del manager
model_manager = ModelManager()
