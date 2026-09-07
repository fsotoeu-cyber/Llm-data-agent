# evals/test_oracle.py
"""Caso B determinista: _ejecutar_operacion_pandas vs oráculo pandas crudo.
Incluye regression tests de los fixes del guard y del validador AST."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from herramientas import _ejecutar_operacion_pandas, resolver_columnas

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
df = pd.read_csv(os.path.join(FIX, "limpio_min.csv"))


def _ejecutar(plan):
    return _ejecutar_operacion_pandas(df, plan, plan.get("agrupar_por"))


def test_promedio_global():
    res = _ejecutar({"operacion": "promedio", "columnas": ["TIEMPO_ENTREGA_MIN"]})
    esperado = round(df["TIEMPO_ENTREGA_MIN"].mean(), 2)
    assert res["TIEMPO_ENTREGA_MIN"] == pytest.approx(esperado, abs=0.01)


def test_conteo_total_sin_columnas():
    """REGRESSION: conteo sin columnas debe funcionar (fix del guard)."""
    res = _ejecutar({"operacion": "conteo", "columnas": []})
    assert res["total_filas"] == len(df)


def test_filtrar_sin_columnas():
    """REGRESSION: filtrar sin columnas debe funcionar (fix del guard)."""
    res = _ejecutar({"operacion": "filtrar", "columnas": [],
                     "filtro": "CLIMA == 'Soleado'"})
    assert res["filas_filtradas"] == (df["CLIMA"] == "Soleado").sum()


def test_filtro_compuesto_bitand():
    """REGRESSION: '&' entre comparaciones no estaba en la whitelist AST
    (BitAnd) — todo filtro compuesto era rechazado. Detectado por la suite,
    no por la batería manual de 31/31."""
    res = _ejecutar({"operacion": "filtrar", "columnas": [],
                     "filtro": "(TIEMPO_ENTREGA_MIN > 30) & (DISTANCIA_KM < 5)"})
    esperado = ((df["TIEMPO_ENTREGA_MIN"] > 30) & (df["DISTANCIA_KM"] < 5)).sum()
    assert res["filas_filtradas"] == esperado


def test_promedio_por_grupo():
    res = _ejecutar({"operacion": "promedio", "columnas": ["TIEMPO_ENTREGA_MIN"],
                     "agrupar_por": ["CLIMA"]})
    oracle = df.groupby("CLIMA")["TIEMPO_ENTREGA_MIN"].mean().round(2).to_dict()
    assert {r["CLIMA"]: r["TIEMPO_ENTREGA_MIN"] for r in res} == pytest.approx(oracle, abs=0.01)


def test_correlacion():
    res = _ejecutar({"operacion": "correlacion",
                     "columnas": ["TIEMPO_ENTREGA_MIN", "DISTANCIA_KM"]})
    esperado = round(df["TIEMPO_ENTREGA_MIN"].corr(df["DISTANCIA_KM"]), 3)
    got = res["TIEMPO_ENTREGA_MIN"]["DISTANCIA_KM"]
    assert got == pytest.approx(esperado, abs=0.001)


def test_resolver_columnas_case_insensitive():
    ok, bad = resolver_columnas(df, ["tiempo_entrega_min", "clima"])
    assert ok == ["TIEMPO_ENTREGA_MIN", "CLIMA"] and bad == []


def test_resolver_columnas_faltantes():
    ok, bad = resolver_columnas(df, ["tiempo_entrega_min", "no_existe"])
    assert ok == ["TIEMPO_ENTREGA_MIN"] and bad == ["no_existe"]
