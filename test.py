#!/usr/bin/env python3
"""
Script de prueba para el servicio Ollama Fake
"""

import requests
import json
import time

BASE_URL = "http://localhost:11436"

def test_health():
    """Probar endpoint de health"""
    print("🏥 Probando health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_list_models():
    """Probar listado de modelos"""
    print("\n📋 Probando listado de modelos...")
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Modelos encontrados: {len(data['models'])}")
        for model in data['models']:
            print(f"  - {model['name']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generate():
    """Probar generación de texto"""
    print("\n🤖 Probando generación de texto...")
    try:
        payload = {
            "model": "qwen2.5:3b",  # Usar el modelo más pequeño para pruebas
            "prompt": "¿Cuál es la capital de España?",
            "stream": False
        }
        response = requests.post(f"{BASE_URL}/api/generate", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Respuesta: {data.get('response', 'No response')}")
        else:
            print(f"Error response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat():
    """Probar chat"""
    print("\n💬 Probando chat...")
    try:
        payload = {
            "model": "qwen2.5:3b",  # Usar el modelo más pequeño para pruebas
            "messages": [
                {"role": "user", "content": "Hola, ¿cómo estás?"}
            ],
            "stream": False
        }
        response = requests.post(f"{BASE_URL}/api/chat", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Respuesta: {data.get('message', {}).get('content', 'No content')}")
        else:
            print(f"Error response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_streaming():
    """Probar streaming"""
    print("\n🌊 Probando streaming...")
    try:
        payload = {
            "model": "qwen2.5:3b",  # Usar el modelo más pequeño para pruebas
            "prompt": "Cuenta hasta cinco",
            "stream": True
        }
        response = requests.post(f"{BASE_URL}/api/generate", json=payload, stream=True)
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return False
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                chunk = data.get('response', '')
                full_response += chunk
                print(chunk, end='', flush=True)
                if data.get('done'):
                    break
        
        print(f"\nRespuesta completa: {full_response}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Ejecutar todas las pruebas"""
    print("🧪 Iniciando pruebas del servicio Ollama Fake...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Listar Modelos", test_list_models),
        ("Generar Texto", test_generate),
        ("Chat", test_chat),
        ("Streaming", test_streaming)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔄 Ejecutando: {test_name}")
        success = test_func()
        results.append((test_name, success))
        print(f"{'✅ PASS' if success else '❌ FAIL'}")
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS:")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nResultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron!")
    else:
        print("⚠️  Algunas pruebas fallaron. Verifica que el servicio esté ejecutándose.")

if __name__ == "__main__":
    main()
