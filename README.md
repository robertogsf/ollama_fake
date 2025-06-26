# Servicio Ollama Fake

Un servicio Python que simula la API de Ollama para ser compatible con n8n y otras aplicaciones que requieren conectarse a Ollama.

## Características

- ✅ Compatible con la API de Ollama
- ✅ Endpoints principales implementados (`/api/generate`, `/api/chat`, `/api/tags`, etc.)
- ✅ Soporte para streaming y respuestas completas
- ✅ Modelos simulados (llama2, codellama, mistral, etc.)
- ✅ Respuestas consistentes basadas en hash del prompt
- ✅ Configuración mediante variables de entorno

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar variables de entorno (opcional):
```bash
cp .env.example .env
# Editar .env según tus necesidades
```

3. Ejecutar el servicio:
```bash
python server.py
```

O usar el script de inicio:
```bash
chmod +x start.sh
./start.sh
```

## Configuración

El servicio se puede configurar mediante variables de entorno en el archivo `.env`:

- `PORT`: Puerto donde se ejecutará (por defecto: 11434)
- `HOST`: Host donde se ejecutará (por defecto: 0.0.0.0)
- `DEBUG`: Modo debug (por defecto: False)
- `DEFAULT_MODEL`: Modelo por defecto (por defecto: llama2)

## Uso con n8n

1. Inicia el servicio Ollama Fake
2. En n8n, usa el nodo "Ollama"
3. Configura la URL base como: `http://localhost:11434`
4. Selecciona cualquiera de los modelos disponibles

## Endpoints Disponibles

### GET /api/tags
Lista todos los modelos disponibles.

### POST /api/generate
Genera texto basado en un prompt.

```json
{
  "model": "llama2",
  "prompt": "¿Cuál es la capital de España?",
  "stream": false
}
```

### POST /api/chat
Chat con formato de mensajes.

```json
{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "Hola, ¿cómo estás?"}
  ],
  "stream": false
}
```

### POST /api/show
Muestra información detallada de un modelo.

### POST /api/pull
Simula la descarga de un modelo.

### POST /api/embeddings
Genera embeddings (simulados) para un texto.

### GET /health
Health check del servicio.

### GET /version
Versión del servicio.

## Modelos Disponibles

- `llama2`
- `llama2:7b`
- `codellama`
- `mistral`

## Pruebas

Puedes probar el servicio con curl:

```bash
# Listar modelos
curl http://localhost:11434/api/tags

# Generar texto
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "prompt": "¿Cuál es la capital de España?"}'

# Chat
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "messages": [{"role": "user", "content": "Hola"}]}'
```

## Desarrollo

Para ejecutar en modo desarrollo:

```bash
export DEBUG=True
python server.py
```

## Notas

- Este es un servicio de simulación para desarrollo y pruebas
- Las respuestas son generadas de forma determinística basada en el hash del prompt
- No se conecta a modelos LLM reales por defecto
- Puedes extender el código para conectar a APIs externas si es necesario
