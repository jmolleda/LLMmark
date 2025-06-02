import sys
import os
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_evaluator import LLMJudge

JSON_PATH = "data/runs/run_1/gemma3:1b/question_1.json"

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
            question=pregunta_texto.strip(),
            expected_answer=respuesta_esperada.strip(),
            model_answer=razonamiento.strip()
        )

        print("\n--- Evaluación ---")
        print("Puntuación:", resultado.get("grade"))
        print("Justificación:", resultado.get("justification"))
        print("Latencia:", resultado.get("latency"), "segundos")
        resultados.append(resultado)

    return resultados

if __name__ == "__main__":
    datos = cargar_resultados(JSON_PATH)

    # Ignorar estadísticas agregadas si existen al principio
    if "num_correct" in datos[0]:
        datos = datos[1:]

    preguntas_por_grupo = agrupar_por_pregunta(datos)

    juez = LLMJudge()
    evaluar_preguntas(juez, preguntas_por_grupo)
