# evals/test_quality_gate.py
"""Caso D determinista: clasificación de fixtures."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from herramientas import validar_dataset

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.mark.parametrize("fixture,esperado,score_min", [
    ("limpio_min.csv", "ok", 95),
    ("mayusculas.csv", "ok", 95),
    ("con_nulos.csv", "warning", 0),
    ("vacio_casi.csv", "bloqueante", 0),
])
def test_clasificacion(fixture, esperado, score_min):
    v = validar_dataset(pd.read_csv(os.path.join(FIX, fixture)))
    got = "bloqueante" if v["bloqueante"] else ("warning" if v["advertencias"] else "ok")
    assert got == esperado
    assert v["score"] >= score_min
    assert 0 <= v["score"] <= 100


def test_dataset_vacio():
    v = validar_dataset(pd.DataFrame())
    assert v["bloqueante"] and v["score"] == 0
