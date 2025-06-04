import os
import json
from collections import defaultdict

from llmmark.evaluation.llm_evaluator import LLMJudge

JSON_PATH = "../data/runs/run_1/gemma3:1b/question_1.json"

def cargar_resultados(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def agrupar_por_pregunta(data):
    preguntas = defaultdict(list)
    for item in data:
        if "question" in item:
            preguntas[item["question"]].append(item)
    return preguntas

def evaluar_preguntas(juez, preguntas_agrupadas):
    resultados = []

    for pregunta_texto, ejecuciones in preguntas_agrupadas.items():
        primera_ejecucion = ejecuciones[0]
        respuesta_esperada = primera_ejecucion.get("correct_answer", "")
        razonamiento = primera_ejecucion.get("raw_answer", "")

        print("\n===============================")
        print("Pregunta:", pregunta_texto.strip())
        print("Respuesta esperada:", respuesta_esperada)
        print("Respuesta del modelo (raw):", razonamiento)

        resultado = juez.eval(
            pregunta=pregunta_texto.strip(),
            respuesta_esperada=respuesta_esperada.strip(),
            respuesta_modelo=razonamiento.strip()
        )

        print("\n--- Evaluación ---")
        print("Puntuación:", resultado.get("grade"))
        print("Justificación:", resultado.get("justificación"))
        print("Latencia:", resultado.get("latencia"), "segundos")
        resultados.append(resultado)

    return resultados

if __name__ == "__main__":
    datos = cargar_resultados(JSON_PATH)

    # Ignorar estadísticas agregadas si existen al principio
    if "num_correct" in datos[0]:
        datos = datos[1:]

    preguntas_por_grupo = agrupar_por_pregunta(datos)

    juez = LLMJudge(modelo="models/gemini-1.5-flash")
    evaluar_preguntas(juez, preguntas_por_grupo)
