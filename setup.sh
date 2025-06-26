#!/bin/bash

echo "🚀 Configurando Ollama con IA Real..."

# Crear directorios para modelos
echo "📁 Creando directorios..."
mkdir -p /home/models/cache
mkdir -p /home/models/hf_cache

# Verificar si ya existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "🐍 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

echo "✅ Setup completado!"
echo ""
echo "Para iniciar el servidor:"
echo "1. source venv/bin/activate"
echo "2. python server_real.py"
echo ""
echo "Para probar:"
echo "python test.py"
echo ""
echo "⚠️  NOTA: El primer uso de un modelo tomará tiempo en descargar (~7-14GB)"
