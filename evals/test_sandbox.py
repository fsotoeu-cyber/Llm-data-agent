# evals/test_sandbox.py
"""Sandbox AST + filtros + timeout. Cero LLM, corre en <5s."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from herramientas import es_codigo_seguro, sanitizar_filtro, _exec_restringido

PLOT_OK = "fig, ax = plt.subplots(figsize=(10, 6))\nax.bar(['a'], [1])"

SEGUROS = [
    PLOT_OK,
    "import pandas as pd",                      # benigno: se elimina luego
    "sns.heatmap(df.corr(), annot=True)",
    "df.groupby('A')['B'].mean().plot(kind='bar')",
]

INSEGUROS = [
    ("import os", "Import prohibido"),
    ("import subprocess", "Import prohibido"),
    ("from pathlib import Path", "ImportFrom prohibido"),
    ("pd.read_csv('.streamlit/secrets.toml')", "read_csv"),
    ("df.to_csv('salida.csv')", "to_csv"),
    ("plt.savefig('/tmp/x.png')", "savefig"),
    ("np.load('x.npy')", "load"),
    ("sns.load_dataset('tips')", "load_dataset"),
    ("open('secrets.txt')", "open"),
    ("while True:\n    pass", "while"),
    ("eval('1+1')", "eval"),
    ("df.__class__", "Atributo prohibido"),
]


@pytest.mark.parametrize("codigo", SEGUROS, ids=lambda c: c[:40])
def test_codigo_seguro(codigo):
    ok, motivo = es_codigo_seguro(codigo)
    assert ok, f"debería ser seguro: {motivo}"


@pytest.mark.parametrize("codigo,frag", INSEGUROS, ids=lambda x: x[:40] if isinstance(x, str) else x[1])
def test_codigo_inseguro(codigo, frag):
    ok, motivo = es_codigo_seguro(codigo)
    assert not ok
    assert frag.lower() in motivo.lower()


# ---- sanitizar_filtro (bitwise & | ~ incluidos desde el fix) ----
@pytest.mark.parametrize("filtro", [
    "CLIMA == 'Soleado'",
    "TIEMPO > 10 & DIST < 5",
    "A in [1, 2]",
    "~(A > 5)",
    "A > 1 | B < 2",
    "(A > 1) & (B < 2) | (C == 'x')",
])
def test_filtro_valido(filtro):
    assert sanitizar_filtro(filtro) == filtro


@pytest.mark.parametrize("filtro", [
    "__import__('os')", "eval('x')", "exec('y')", "lambda x: x",
    "open('f')", "df.query('x')", "f'{x}'", "[y for y in x]",
])
def test_filtro_rechazado(filtro):
    with pytest.raises(ValueError):
        sanitizar_filtro(filtro)


# ---- timeout ----
def test_exec_timeout():
    import time
    t0 = time.time()
    with pytest.raises(TimeoutError):
        _exec_restringido("x = 0\nfor _ in range(10**9): x += 1", {}, timeout_s=2)
    assert time.time() - t0 < 5  # abortó (el hilo residual no bloquea)
