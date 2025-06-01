import os
import time
import json
import google.generativeai as genai

class LLMJudge:
    def __init__(self, api_key=None, modelo="models/gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.modelo = genai.GenerativeModel(modelo)
        self.historial = []

    def generate_prompt(self, pregunta, respuesta_esperada, respuesta_modelo):
        prompt = f"""Eres un evaluador riguroso de exámenes.

        Pregunta:
        {pregunta}

        Respuesta esperada:
        {respuesta_esperada}

        Respuesta generada por el modelo:
        {respuesta_modelo}

        Evalúa la calidad de la respuesta del modelo en base a:
        - Corrección
        - Relevancia
        - Claridad

        Devuelve:
        - Una justificación breve
        - Una puntuación del 0.0 (muy mal) al 1.0 (perfecta)

        Responde en formato JSON así:
        {{"grade": "0.0", "justificación": "..."}}"""
        return prompt

    def eval(self, pregunta, respuesta_esperada, respuesta_modelo):
        prompt = self.generate_prompt(pregunta, respuesta_esperada, respuesta_modelo)
        inicio = time.time()
        respuesta = self.modelo.generate_content(prompt)
        latencia = round(time.time() - inicio, 3)

        try:
            resultado = json.loads(respuesta.text)
        except json.JSONDecodeError as e:
            print("Error al decodificar la respuesta JSON.", e)
            resultado = {
                "grade": "",
                "justificación": f"Formato de respuesta no válido:\n{respuesta.text}"
            }
        resultado["latencia"] = latencia
        self.historial.append(resultado)
        return resultado
