# evals/run_eval.py
"""Suite de evaluación reproducible (A/B/C/D del README). v2.1

Uso:
    GROQ_API_KEY=gsk_... LANGCHAIN_API_KEY=... python evals/run_eval.py
    python evals/run_eval.py --tipos D                  # solo Quality Gate (sin LLM)
    python evals/run_eval.py --tipos B --estricto       # match debe ser el 1er número
    python evals/run_eval.py --report evals/otro.json   # no pisar el reporte principal

v2 (rigor + telemetría):
- tool_selection: PASS = "eventualmente correcta", pero cada caso A registra
  tool_first_valid_attempt_ok / exception_count / attempts / final_attempt.
- groupby (B): exige TODOS los grupos del oráculo, matching uno a uno.
- cálculo escalar (B): default any(); --estricto exige match en el primer número.
v2.1 (resiliencia de cuota):
- Reporte incremental: un 429 a mitad de suite no pierde lo corrido.
- Backoff 45s + reintento por caso ante 429.
- n_casos imprime como número (no como porcentaje).
- retry_rate real sigue N/A (API de traces de LangSmith); proxy = _Exception.
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE))

from herramientas import validar_dataset, analizar_datos  # noqa: E402

from agent_helper import preparar_session_state, construir_agente, get_llm  # noqa: E402

FIXTURES = os.path.join(BASE, "fixtures")


# ---------------------------------------------------------------------
# ORÁCULO PANDAS
# ---------------------------------------------------------------------
def resolver_oracle(df: pd.DataFrame, spec: str):
    """'oracle:mean:COL' | 'oracle:len' | 'oracle:groupby_mean:COL:GRUPO'
    | 'oracle:corr:COL1:COL2' → valor(es) calculados con pandas puro."""
    partes = spec.split(":")
    kind = partes[1]
    if kind == "len":
        return float(len(df))
    if kind == "mean":
        return float(df[partes[2]].mean())
    if kind == "corr":
        return float(df[partes[2]].corr(df[partes[3]]))
    if kind == "groupby_mean":
        col, grupo = partes[2], partes[3]
        serie = df.groupby(grupo)[col].mean()
        return {str(k): float(v) for k, v in serie.items()}
    raise ValueError(f"Oráculo desconocido: {spec}")


def extraer_numeros(texto: str) -> list[float]:
    """Extrae números del texto del LLM (normaliza guiones Unicode con
    significado negativo — U+2010–U+2014, U+2212 —, coma decimal y signo
    separado por espacio)."""
    for d in [chr(c) for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212)]:
        texto = texto.replace(d, "-")
    texto = texto.replace(",", ".")
    texto = re.sub(r"(-)\s+(\d)", r"\1\2", texto)
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", texto)]


# ---------------------------------------------------------------------
# COMPARACIÓN
# ---------------------------------------------------------------------
def _tol_check(o: float, e: float, caso: dict) -> bool:
    tol_rel = caso.get("tolerancia_relativa", 0.01)
    tol_abs = caso.get("tolerancia_absoluta", abs(e) * tol_rel)
    return abs(o - e) <= tol_abs


def comparar(esperado, obtenidos: list[float], caso: dict,
             estricto: bool = False) -> tuple[bool, dict]:
    """Devuelve (ok, evidencia) — la evidencia va al reporte.

    - dict (groupby): TODOS los valores del oráculo deben aparecer, con
      matching uno a uno: cada valor esperado CONSUME un número distinto
      de la respuesta. {'a': 1, 'b': 1} vs [1.0] → False.
    - escalar: any() por defecto. Con estricto=True, el número coincidente
      debe ser el PRIMERO de la respuesta.
    """
    if isinstance(esperado, dict):
        usados = [False] * len(obtenidos)
        matched = {}
        for grupo, val in esperado.items():
            hit = None
            for i, o in enumerate(obtenidos):
                if not usados[i] and abs(o - val) <= abs(val) * caso.get("tolerancia_relativa", 0.01):
                    hit = i
                    break
            if hit is None:
                return False, {
                    "grupos_esperados": len(esperado),
                    "grupos_matcheados": len(matched),
                    "grupo_faltante": grupo,
                    "valor_faltante": val,
                    "numeros": obtenidos,
                }
            usados[hit] = True
            matched[grupo] = obtenidos[hit]
        return True, {"grupos_matcheados": matched}

    for i, o in enumerate(obtenidos):
        if _tol_check(o, esperado, caso):
            if estricto and i != 0:
                return False, {
                    "motivo": "match no es el primer número (modo estricto)",
                    "primer_numero": obtenidos[0],
                    "match_idx": i,
                    "esperado": esperado,
                }
            return True, {"match_idx": i, "esperado": esperado}
    return False, {"esperado": esperado, "numeros": obtenidos}


# ---------------------------------------------------------------------
# TELEMETRÍA DE AGENTE
# ---------------------------------------------------------------------
def _telemetria_tool(tools_llamadas: list, esperada: str | None = None) -> dict:
    """attempts / exception_count / first_valid / final de los
    intermediate_steps. _Exception = error de parseo ReAct absorbido
    por handle_parsing_errors."""
    reales = [t for t in tools_llamadas if t != "_Exception"]
    tele = {
        "attempts": len(tools_llamadas),
        "exception_count": sum(1 for t in tools_llamadas if t == "_Exception"),
        "first_valid_tool": reales[0] if reales else None,
        "final_attempt": tools_llamadas[-1] if tools_llamadas else None,
    }
    if esperada is not None:
        tele["tool_first_valid_attempt_ok"] = tele["first_valid_tool"] == esperada
    return tele


# ---------------------------------------------------------------------
# EJECUTORES POR TIPO
# ---------------------------------------------------------------------
def eval_d(caso: dict) -> dict:
    df = pd.read_csv(os.path.join(FIXTURES, caso["fixture"]))
    v = validar_dataset(df)
    esp = caso["clasificacion_esperada"]
    got = "bloqueante" if v["bloqueante"] else ("warning" if v["advertencias"] else "ok")
    ok = got == esp
    if ok and "score_min" in caso:
        ok = v["score"] >= caso["score_min"]
    return {"ok": ok, "detalle": f"esperado={esp} obtenido={got} score={v['score']}"}


def eval_b(caso: dict, llm, estricto: bool = False) -> dict:
    import streamlit as st
    preparar_session_state(os.path.join(FIXTURES, caso["fixture"]))
    st.session_state["llm"] = llm
    salida = analizar_datos(caso["pregunta"])  # nivel tool: sin agente
    df = pd.read_csv(os.path.join(FIXTURES, caso["fixture"]))
    esperado = resolver_oracle(df, caso["esperado"])
    numeros = extraer_numeros(salida)
    ok, evid = comparar(esperado, numeros, caso, estricto=estricto)
    tele = {  # nivel tool: una sola invocación, sin pasos intermedios
        "attempts": 1, "exception_count": 0,
        "first_valid_tool": "Analizar Datos", "final_attempt": "Analizar Datos",
        "tool_first_valid_attempt_ok": True,
    }
    return {
        "ok": ok,
        "detalle": f"oracle={esperado} | evidencia={evid} | salida={salida[:120]}",
        "numeros_extraidos": numeros[:10],
        **tele,
    }


def eval_a_c(caso: dict, agente) -> dict:
    preparar_session_state(os.path.join(FIXTURES, caso["fixture"]))
    resp = agente.invoke({"input": caso["pregunta"]})
    out = str(resp.get("output", ""))
    steps = resp.get("intermediate_steps", [])
    tools_llamadas = [a.tool for a, _ in steps if hasattr(a, "tool")]

    tele = _telemetria_tool(tools_llamadas, caso.get("tool_esperada"))

    checks = []
    if "tool_esperada" in caso:
        ok_tool = caso["tool_esperada"] in tools_llamadas  # criterio principal: sin cambio
        checks.append((ok_tool, f"tools={tools_llamadas}"))
        if not ok_tool:
            return {"ok": False, "detalle": f"tool esperada={caso['tool_esperada']} llamadas={tools_llamadas}", **tele}
    for frag in caso.get("salida_debe_contener", []):
        checks.append((frag.lower() in out.lower(), f"debe contener '{frag}'"))
    cualquiera = caso.get("salida_debe_contener_cualquiera", [])
    if cualquiera:
        ok_any = any(f.lower() in out.lower() for f in cualquiera)
        checks.append((ok_any, f"debe contener alguna de: {cualquiera}"))
    for frag in caso.get("salida_no_debe_contener", []):
        checks.append((frag.lower() not in out.lower(), f"no debe contener '{frag}'"))
    ok = all(c for c, _ in checks)
    fallidos = [d for c, d in checks if not c]
    return {"ok": ok, "detalle": "; ".join(fallidos) or f"tools={tools_llamadas} ok", **tele}


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipos", default="A,B,C,D", help="subconjunto, ej. 'D' o 'A,C'")
    ap.add_argument("--casos", default=os.path.join(BASE, "casos.json"))
    ap.add_argument("--report", default=os.path.join(BASE, "reporte_eval.json"))
    ap.add_argument("--estricto", action="store_true",
                    help="cálculos escalares: el match debe ser el PRIMER número de la respuesta")
    args = ap.parse_args()
    tipos = {t.strip().upper() for t in args.tipos.split(",")}

    casos = json.load(open(args.casos, encoding="utf-8"))["casos"]
    casos = [c for c in casos if c["tipo"] in tipos]
    if not casos:
        print("Sin casos para los tipos pedidos.")
        return

    llm = agente = None
    if any(c["tipo"] in ("A", "C") for c in casos):
        if not os.environ.get("GROQ_API_KEY"):
            print("❌ GROQ_API_KEY requerida para tipos A/C. Usa --tipos B,D para suite sin LLM.")
            sys.exit(1)
        llm = get_llm()
        agente = construir_agente(llm)
    if any(c["tipo"] == "B" for c in casos):
        if not os.environ.get("GROQ_API_KEY"):
            print("❌ GROQ_API_KEY requerida para tipo B (analizar_datos usa LLM). Usa --tipos D.")
            sys.exit(1)
        if llm is None:
            llm = get_llm()

    resultados = []
    for c in casos:
        t0 = time.time()
        try:
            if c["tipo"] == "D":
                r = eval_d(c)
            elif c["tipo"] == "B":
                r = eval_b(c, llm, estricto=args.estricto)
            else:
                r = eval_a_c(c, agente)
        except Exception as e:
            msg = str(e)
            # v2.1: backoff 45s + reintento por caso ante 429
            if ("rate limit" in msg.lower() or "429" in msg) and c["tipo"] != "D":
                espera = 45
                print(f"⏳ 429 en {c['id']}: esperando {espera}s y reintentando...")
                time.sleep(espera)
                try:
                    if c["tipo"] == "B":
                        r = eval_b(c, llm, estricto=args.estricto)
                    else:
                        r = eval_a_c(c, agente)
                except Exception as e2:
                    r = {"ok": False, "detalle": f"excepción tras reintento: {e2}"}
            else:
                r = {"ok": False, "detalle": f"excepción: {e}"}
        r["segundos"] = round(time.time() - t0, 1)
        resultados.append({"id": c["id"], "tipo": c["tipo"], **r})
        # v2.1: persistencia incremental — la muerte por cuota no pierde lo corrido
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"metricas": "parcial — run interrumpido", "resultados": resultados},
                      f, ensure_ascii=False, indent=2)
        mark = "✅" if r["ok"] else "❌"
        extra = ""
        if "exception_count" in r and r["exception_count"]:
            extra = f" [⚠ {r['exception_count']} exc, first_valid={r.get('first_valid_tool')}]"
        print(f"{mark} {c['id']} [{c['tipo']}] ({r['segundos']}s){extra} — {r['detalle'][:110]}")

    # ------- Métricas -------
    def rate(tipo):
        sub = [r for r in resultados if r["tipo"] == tipo]
        return (sum(r["ok"] for r in sub) / len(sub) * 100) if sub else None

    sub_first = [r for r in resultados
                 if r["tipo"] == "A" and "tool_first_valid_attempt_ok" in r]
    first_valid_rate = (sum(r["tool_first_valid_attempt_ok"] for r in sub_first)
                        / len(sub_first) * 100) if sub_first else None

    sub_agent = [r for r in resultados if r.get("exception_count") is not None]
    con_exc = sum(1 for r in sub_agent if r["exception_count"] > 0)
    exc_total = sum(r["exception_count"] for r in sub_agent)
    retry_proxy = (con_exc / len(sub_agent) * 100) if sub_agent else None

    metricas = {
        "tool_selection_accuracy": rate("A"),
        "tool_first_valid_attempt_rate": first_valid_rate,
        "calculation_accuracy": rate("B"),
        "error_handling_rate": rate("C"),
        "pass_rate": sum(r["ok"] for r in resultados) / len(resultados) * 100,
        "retry_rate": "N/A (requiere integración con API de traces de LangSmith)",
        "retry_rate_proxy": (f"{retry_proxy:.0f}% ({con_exc}/{len(sub_agent)} casos, "
                             f"{exc_total} _Exception totales)") if retry_proxy is not None else None,
        "n_casos": len(resultados),
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    print("\n===== MÉTRICAS =====")
    for k, v in metricas.items():
        if isinstance(v, str) or v is None or k == "n_casos":
            print(f"{k:32s} {v if v is not None else '—'}")
        else:
            print(f"{k:32s} {v:.0f}%")

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"metricas": metricas, "resultados": resultados}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n📄 Reporte: {args.report}")


if __name__ == "__main__":
    main()
