# evals/test_agent.py
"""Casos A/C con LLM real. Skip automático sin GROQ_API_KEY."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"), reason="requiere GROQ_API_KEY (LLM real)"
)

from agent_helper import preparar_session_state, construir_agente, get_llm  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
_llm = get_llm()


@pytest.fixture
def agente():
    return construir_agente(_llm)


CASOS_A = [
    ("limpio_min.csv", "¿Cuál es el promedio de TIEMPO_ENTREGA_MIN?", "Analizar Datos"),
    ("limpio_min.csv", "Genera un gráfico de barras del tiempo promedio por clima", "Generar Gráfico"),
    ("limpio_min.csv", "Dame la información general del dataset", "Información DF"),
    ("limpio_min.csv", "Estadísticas de las variables numéricas", "Resumen Estadístico"),
    ("con_nulos.csv", "Limpia los datos", "Limpiar Datos"),
]


@pytest.mark.parametrize("fixture,pregunta,tool", CASOS_A)
def test_tool_selection(agente, fixture, pregunta, tool):
    preparar_session_state(os.path.join(FIX, fixture))
    st = agente.invoke({"input": pregunta})
    tools_llamadas = [a.tool for a, _ in st.get("intermediate_steps", []) if hasattr(a, "tool")]
    assert tool in tools_llamadas, f"esperada={tool} llamadas={tools_llamadas}"


def test_columna_inexistente_softfail(agente):
    preparar_session_state(os.path.join(FIX, "limpio_min.csv"))
    st = agente.invoke({"input": "Promedio de columna_que_no_existe"})
    out = str(st.get("output", ""))
    assert "Traceback" not in out
    assert ("no encontrada" in out.lower()) or ("disponibles" in out.lower())
