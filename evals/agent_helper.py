# evals/agent_helper.py
"""Construye el agente para evals.

DIVERGENCIAS deliberadas con app.py:
- Sin ConversationBufferMemory → cada caso corre aislado (reproducibilidad).
- Prompt sin {chat_history} (no hay memoria que llenar).
- Prompt sin muestra del dataset (el eval mide routing, no contexto).
  NOTA: el prompt de routing está DUPLICADO con app.py — cualquier cambio
  de reglas debe aplicarse en AMBOS (v2: regla de dominancia gráficos).

TODO (refactor item 5): extraer build_agent() y el prompt a un módulo
propio para eliminar esta duplicación.
"""
import os
import sys
import warnings

import pandas as pd
import streamlit as st

# Silencia warnings de streamlit en bare mode (sin `streamlit run`)
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor

from herramientas import crear_herramientas

# Copia del prompt de routing de app.py, SIN chat_history y SIN muestra.
# v2: regla de dominancia de gráficos (fix A2) — alineada con app.py.
PROMPT_EVAL = """Eres un analista de datos. Sigue el formato ReAct.

Herramientas:
{tools}

Nombres EXACTOS: [{tool_names}]

REGLAS:
- info/nulos/duplicados (todo el DF) → Información DF
- estadísticas de todas las numéricas → Resumen Estadístico
- promedio/suma/conteo/max/min/correlación/outliers de columnas concretas → Analizar Datos
- Si la pregunta pide gráfico/visualización/barras/heatmap/plot → Generar Gráfico SIEMPRE, aunque también mencione promedio/suma/estadística (la visualización DOMINA sobre el cálculo)
- limpiar → Limpiar Datos
- Si piden promedio/suma/conteo/max/min de UNA columna (exista o no) → Analizar Datos (NUNCA Información DF)
- Si el mensaje es solo un nombre raro de columna o "columna_que_no_existe" → Analizar Datos

Formato:
Thought: breve
Action: nombre_exacto
Action Input: pregunta del usuario sin reescribir
Observation: (sistema)
Thought: listo
Final Answer: en español

NUNCA respondas sin herramienta.
NUNCA omitas Action / Action Input.

Question: {input}
Thought: {agent_scratchpad}
"""


def get_llm():
    return ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model_name="openai/gpt-oss-120b",
        temperature=0,
    )


def preparar_session_state(fixture_path: str) -> pd.DataFrame:
    """Carga un fixture y lo inyecta en session_state (bare mode)."""
    df = pd.read_csv(fixture_path)
    # Reset de claves que usan los tools
    for k in ["df_original", "df_clean", "log_limpieza", "cleaned",
              "validacion_post", "graficos_generados"]:
        st.session_state.pop(k, None)
    st.session_state["df_original"] = df.copy()
    st.session_state["df_clean"] = df.copy()
    st.session_state["log_limpieza"] = []
    st.session_state["cleaned"] = False
    st.session_state["graficos_generados"] = []
    return df


def construir_agente(llm) -> AgentExecutor:
    tools = crear_herramientas(llm)
    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=PROMPT_EVAL,
    )
    agente = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agente,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=90,
        return_intermediate_steps=True,  # ← expone la tool elegida sin parsear texto
    )
