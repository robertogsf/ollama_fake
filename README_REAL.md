# Ollama con IA Real

Este proyecto simula la API de Ollama pero usando modelos de IA reales con Hugging Face Transformers, incluyendo el modelo **Qwen2.5:14B**.

## 🎯 Características

- ✅ API compatible con Ollama
- 🤖 Modelos de IA reales (Qwen2.5, Llama3.2)
- 🔄 Streaming de respuestas
- 📊 Endpoints completos (/api/generate, /api/chat, /api/tags, etc.)
- 🎮 Descarga automática de modelos
- 🔧 Gestión inteligente de memoria GPU/CPU

## 🚀 Instalación Rápida

```bash
# Clonar y configurar
cd ollama_con_poderes
./setup.sh

# Activar entorno
source venv/bin/activate

# Iniciar servidor
python server_real.py
```

## 🤖 Modelos Disponibles

| Modelo | Tamaño | Descripción |
|--------|--------|-------------|
| `qwen2.5:14b` | ~28GB | Qwen2.5 14B Instruct (Principal) |
| `qwen2.5:7b` | ~14GB | Qwen2.5 7B Instruct (Recomendado) |
| `qwen2.5:3b` | ~6GB | Qwen2.5 3B Instruct (Eficiente) |
| `llama3.2:3b` | ~6GB | Llama 3.2 3B Instruct |

## 📡 Endpoints API

### Listar modelos
```bash
curl http://localhost:11434/api/tags
```

### Generar texto
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "prompt": "¿Cuál es la capital de España?",
    "stream": false
  }'
```

### Chat
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [
      {"role": "user", "content": "Hola, ¿cómo estás?"}
    ],
    "stream": false
  }'
```

### Descargar modelo
```bash
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "qwen2.5:7b"}'
```

## 🧪 Pruebas

```bash
# Ejecutar suite de pruebas
python test.py

# Probar modelo específico
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:14b", "prompt": "Escribe un poema corto"}'
```

## ⚙️ Configuración

Edita `.env` para personalizar:

```env
# Puerto del servidor
PORT=11434

# Modelo por defecto
DEFAULT_MODEL=qwen2.5:7b

# Cache de modelos
TRANSFORMERS_CACHE=/home/models/cache
HF_HOME=/home/models/hf_cache
```

## 🔧 Uso con n8n

1. Configura n8n para usar `http://localhost:11434` como URL de Ollama
2. Los modelos aparecerán como `qwen2.5:7b`, `qwen2.5:14b`, etc.
3. El primer uso descargará el modelo automáticamente

## 📋 Requisitos del Sistema

- **RAM**: Mínimo 16GB (32GB recomendado para modelos grandes)
- **GPU**: NVIDIA con CUDA (opcional pero recomendado)
- **Espacio**: 30-50GB para caché de modelos
- **Python**: 3.8+

## 🚨 Notas Importantes

- ⏳ **Primera ejecución**: Descarga del modelo puede tomar 10-30 minutos
- 💾 **Memoria**: Modelos grandes requieren mucha RAM/VRAM
- 🔄 **Cambio de modelo**: Descarga automáticamente el modelo nuevo
- 🗑️ **Limpieza**: Usa `/api/models/unload` para liberar memoria

## 🐛 Troubleshooting

### Error de memoria
```bash
# Usar modelo más pequeño
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "qwen2.5:3b", "prompt": "test"}'

# Descargar modelo actual
curl -X POST http://localhost:11434/api/models/unload
```

### Error de CUDA
```bash
# Forzar uso de CPU en model_config.py
DEVICE_CONFIG = {
    "auto_device_map": False,  # Cambiar a False
    ...
}
```

## 📈 Monitoreo

```bash
# Health check
curl http://localhost:11434/health

# Estado del servidor
curl http://localhost:11434/version
```

## 🔗 Integración con Docker

Para usar en Docker con n8n, expone el puerto 11434 y configura las variables de entorno apropiadas.

---

**¡Disfruta de tu IA real con compatibilidad Ollama!** 🎉
