#!/bin/bash

# Script de instalación completa para Ollama Fake Service

set -e  # Salir si hay errores

echo "🚀 Instalación de Ollama Fake Service"
echo "======================================"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado. Por favor instálalo primero."
    exit 1
fi

echo "✅ Python3 encontrado: $(python3 --version)"

# Crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv .venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source .venv/bin/activate

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Hacer scripts ejecutables
echo "🔐 Configurando permisos..."
chmod +x start.sh
chmod +x test.py

# Verificar instalación
echo "🧪 Verificando instalación..."
python -c "import flask, flask_cors, requests; print('✅ Todas las dependencias instaladas correctamente')"

echo ""
echo "🎉 ¡Instalación completada!"
echo ""
echo "📋 Comandos disponibles:"
echo "  ./start.sh                 - Iniciar el servicio"
echo "  python test.py             - Ejecutar pruebas"
echo "  python server.py           - Iniciar directamente"
echo ""
echo "🔗 URLs importantes:"
echo "  Servicio: http://localhost:11434"
echo "  Health:   http://localhost:11434/health"
echo "  Modelos:  http://localhost:11434/api/tags"
echo ""
echo "📖 Para más información, ver:"
echo "  - README.md (documentación general)"
echo "  - N8N_SETUP.md (configuración específica para n8n)"
