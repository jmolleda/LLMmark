import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_evaluator import LLMJudge

if __name__ == "__main__":
    pregunta = "¿Cuál es la capital de Alemania?"
    respuesta_esperada = "[B] Berlín"
    respuesta_modelo = "La capital de Alemania es Berlín"
    juez = LLMJudge(modelo="models/gemini-1.5-flash")
    resultado = juez.eval(pregunta, respuesta_esperada, respuesta_modelo)

    print("\n=== Resultado de evaluación ===")
    print("Puntuación:", resultado.get("puntuación"))
    print("Justificación:", resultado.get("justificación"))
    print("Latencia:", resultado.get("latencia"), "segundos")
