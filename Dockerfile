FROM python:3.11-slim

WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY server.py .
COPY .env .

# Exponer puerto
EXPOSE 11434

# Variables de entorno por defecto
ENV PORT=11434
ENV HOST=0.0.0.0
ENV DEBUG=False

# Comando para ejecutar la aplicación
CMD ["python", "server.py"]
