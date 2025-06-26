#!/usr/bin/env python3
"""
Script para probar la conectividad desde n8n
"""

import requests
import json

# Configuración - ajusta la IP según tu setup
BASE_URL = "http://192.168.1.34:11436"  # IP del servidor donde corre el servicio

def test_basic_connection():
    """Probar conexión básica"""
    print("🔍 Probando conexión básica...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_models_list():
    """Probar listado de modelos"""
    print("\n📋 Probando listado de modelos...")
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Modelos: {[m['name'] for m in data['models']]}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_simple_generation():
    """Probar generación simple"""
    print("\n🤖 Probando generación simple...")
    try:
        payload = {
            "model": "qwen2.5:3b",
            "prompt": "Hola, responde con una sola palabra: Buenos",
            "stream": False
        }
        response = requests.post(f"{BASE_URL}/api/generate", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Respuesta: {data.get('response', 'No response')}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Probando conectividad desde perspectiva de n8n...")
    print(f"🌐 URL base: {BASE_URL}")
    print("=" * 50)
    
    tests = [
        ("Conexión básica", test_basic_connection),
        ("Listado de modelos", test_models_list),
        ("Generación simple", test_simple_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
        print(f"{'✅ PASS' if success else '❌ FAIL'}")
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nResultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡La conexión desde n8n debería funcionar!")
    else:
        print("⚠️  Hay problemas de conectividad.")
        print("💡 Verifica:")
        print("   - Que el servidor esté ejecutándose")
        print("   - Que la IP/puerto sean correctos en n8n")
        print("   - Que no haya firewall bloqueando")
