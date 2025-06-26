#!/bin/bash

# Script para iniciar el servicio Ollama Fake

echo "🔧 Instalando dependencias..."
pip install -r requirements.txt

echo "🚀 Iniciando servicio Ollama Fake..."
python server.py
