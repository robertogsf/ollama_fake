# Configuración para n8n

## Pasos para conectar n8n con el servicio Ollama Fake

### 1. Iniciar el servicio

```bash
# Opción 1: Usando Python directamente
cd /home/rober/Documents/Dazlabs/ollama_fake
source .venv/bin/activate  # Si usas entorno virtual
python server.py

# Opción 2: Usando el script de inicio
./start.sh

# Opción 3: Usando Docker
docker-compose up -d
```

### 2. Verificar que el servicio esté funcionando

```bash
curl http://localhost:11434/health
```

Deberías recibir una respuesta como:
```json
{
  "status": "healthy",
  "service": "ollama-fake",
  "timestamp": "2025-06-26T13:49:12.447724Z"
}
```

### 3. Configurar n8n

1. **Abrir n8n** en tu navegador
2. **Crear un nuevo workflow**
3. **Agregar un nodo Ollama**:
   - Buscar "Ollama" en la lista de nodos
   - Arrastrar el nodo al canvas

4. **Configurar la conexión**:
   - **Base URL**: `http://localhost:11434` (o el puerto que hayas configurado)
   - **Model**: Seleccionar uno de los disponibles:
     - `llama2`
     - `llama2:7b`
     - `codellama`
     - `mistral`

5. **Configurar el prompt**:
   - En el campo "Prompt" o "Message", escribir tu consulta
   - Ejemplo: "¿Cuál es la capital de España?"

### 4. Ejemplo de configuración en n8n

```json
{
  "nodes": [
    {
      "parameters": {
        "baseURL": "http://localhost:11434",
        "model": "llama2",
        "prompt": "Explica qué es la inteligencia artificial en términos simples",
        "options": {}
      },
      "type": "n8n-nodes-base.ollama",
      "typeVersion": 1,
      "position": [600, 300],
      "id": "ollama-node",
      "name": "Ollama"
    }
  ]
}
```

### 5. Verificar conexión

Una vez configurado el nodo:
1. **Ejecutar el nodo** haciendo clic en "Execute node"
2. **Verificar la respuesta** en el panel de salida
3. La respuesta debería contener el texto generado por el modelo simulado

### 6. Solución de problemas comunes

#### Error de conexión
- Verificar que el servicio esté ejecutándose: `curl http://localhost:11434/health`
- Verificar que el puerto sea correcto
- Verificar que no haya firewall bloqueando la conexión

#### Modelo no encontrado
- Verificar que el modelo esté en la lista de modelos disponibles
- Usar: `curl http://localhost:11434/api/tags` para ver modelos disponibles

#### Respuestas lentas
- Es normal en modo desarrollo, las respuestas incluyen delays simulados
- Para desarrollo más rápido, puedes modificar los `time.sleep()` en `server.py`

### 7. Personalización

Puedes personalizar las respuestas editando la función `generate_fake_response()` en `server.py`:

```python
def generate_fake_response(prompt, model="llama2"):
    # Personalizar respuestas según el prompt o modelo
    if "capital" in prompt.lower():
        return "Madrid es la capital de España."
    elif "tiempo" in prompt.lower():
        return "Hoy hace buen tiempo."
    # ... más personalizaciones
    
    # Respuesta por defecto
    return f"Respuesta simulada para: {prompt}"
```

### 8. Uso en producción

Para usar en producción, considera:
- Usar un servidor WSGI como Gunicorn: `gunicorn -w 4 -b 0.0.0.0:11434 server:app`
- Configurar un proxy inverso con Nginx
- Usar Docker para contenerización
- Configurar monitoreo y logs

### 9. Variables de entorno importantes

```bash
# Puerto del servicio (default: 11434 para compatibilidad con Ollama)
PORT=11434

# Host (default: 0.0.0.0 para acceso desde cualquier IP)
HOST=0.0.0.0

# Modelo por defecto
DEFAULT_MODEL=llama2

# Debug mode
DEBUG=false
```

### 10. API Endpoints disponibles para n8n

- `GET /api/tags` - Lista modelos disponibles
- `POST /api/generate` - Genera texto (principal para n8n)
- `POST /api/chat` - Chat con mensajes
- `POST /api/show` - Información del modelo
- `GET /health` - Health check
