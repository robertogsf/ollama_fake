"""
Configuración de modelos con respuestas simuladas para casos de emergencia
"""

# Respuestas predefinidas para cuando no se pueden cargar modelos reales
FALLBACK_RESPONSES = {
    "qwen2.5:7b": [
        "Hola, soy Qwen2.5 7B. ¿En qué puedo ayudarte hoy?",
        "Como modelo de inteligencia artificial, puedo ayudarte con una variedad de tareas.",
        "Estoy aquí para responder tus preguntas y asistirte en lo que necesites.",
        "Gracias por tu pregunta. Basándome en mi entrenamiento, puedo decirte que...",
        "Es un placer poder ayudarte. Mi función es proporcionarte información útil y precisa.",
    ],
    "qwen2.5:3b": [
        "¡Hola! Soy Qwen2.5 3B, aquí para ayudarte.",
        "Perfecto, entiendo tu consulta. Déjame ayudarte con eso.",
        "Como asistente de IA, puedo proporcionarte información sobre diversos temas.",
        "Excelente pregunta. Te puedo ayudar con eso.",
        "Gracias por consultarme. Aquí tienes una respuesta útil.",
    ],
    "qwen2.5:14b": [
        "Soy Qwen2.5 14B, el modelo más avanzado disponible. ¿Cómo puedo asistirte?",
        "Con mi amplio conocimiento, puedo ayudarte con consultas complejas y detalladas.",
        "Como modelo de gran escala, tengo acceso a información extensa para responder tus preguntas.",
        "Mi entrenamiento me permite ofrecer respuestas detalladas y precisas.",
        "Estoy diseñado para manejar consultas sofisticadas. ¿En qué puedo ayudarte?",
    ],
    "llama3.2:3b": [
        "¡Hola! Soy Llama 3.2 3B. Estoy aquí para ayudarte.",
        "Como modelo Llama, puedo asistirte con diversas tareas y preguntas.",
        "Perfecto, entiendo lo que necesitas. Déjame ayudarte.",
        "Mi entrenamiento me permite responder a una amplia gama de consultas.",
        "¡Excelente! Me complace poder ayudarte con tu consulta.",
    ]
}

def get_fallback_response(model_key, prompt):
    """
    Obtener una respuesta de respaldo cuando no se puede cargar el modelo real
    """
    import hashlib
    
    # Usar respuestas del modelo si existe, sino usar genéricas
    responses = FALLBACK_RESPONSES.get(model_key, FALLBACK_RESPONSES["qwen2.5:7b"])
    
    # Seleccionar respuesta basada en hash del prompt para consistencia
    hash_obj = hashlib.md5(prompt.encode())
    index = int(hash_obj.hexdigest(), 16) % len(responses)
    
    base_response = responses[index]
    
    # Si el prompt contiene una pregunta específica, intentar dar una respuesta más contextual
    prompt_lower = prompt.lower()
    
    if "hola" in prompt_lower or "hello" in prompt_lower:
        return "¡Hola! Me da mucho gusto saludarte. ¿En qué puedo ayudarte hoy?"
    elif "nombre" in prompt_lower or "name" in prompt_lower:
        return f"Mi nombre es {model_key}. Soy un modelo de inteligencia artificial diseñado para ayudarte."
    elif "como estas" in prompt_lower or "how are you" in prompt_lower:
        return "¡Estoy muy bien, gracias por preguntar! Como IA, siempre estoy listo para ayudar. ¿Y tú cómo estás?"
    elif "que puedes hacer" in prompt_lower or "what can you do" in prompt_lower:
        return "Puedo ayudarte con muchas cosas: responder preguntas, explicar conceptos, ayudar con tareas de escritura, resolver problemas, y mucho más. ¿Hay algo específico en lo que te gustaría que te ayude?"
    elif "gracias" in prompt_lower or "thanks" in prompt_lower:
        return "¡De nada! Es un placer poder ayudarte. Si tienes más preguntas, no dudes en hacérmelas."
    elif "?" in prompt:
        return f"{base_response} Según mi entrenamiento, esta es una pregunta interesante que requiere consideración cuidadosa."
    else:
        return f"{base_response} Respecto a tu consulta: '{prompt[:100]}...', puedo decirte que es un tema relevante que merece atención."
