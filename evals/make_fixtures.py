# evals/make_fixtures.py
"""Genera los fixtures CSV de la suite. Determinista (seed fija).

Correr desde la raíz del repo:
    python evals/make_fixtures.py
"""
import os
import numpy as np
import pandas as pd

SEED = 42
OUT = os.path.join(os.path.dirname(__file__), "fixtures")


def limpio_min() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 20
    return pd.DataFrame({
        "ID_PEDIDO": range(1001, 1001 + n),
        "CLIMA": (["Soleado", "Lluvioso", "Nublado"] * ((n // 3) + 1))[:n],
        "TIEMPO_ENTREGA_MIN": rng.integers(20, 70, n),
        "DISTANCIA_KM": np.round(rng.uniform(1.0, 9.0, n), 2),
        "FECHA_PEDIDO": pd.date_range("2025-01-01", periods=n, freq="12h"),
    })


def mayusculas() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    n = 12
    return pd.DataFrame({
        "TIEMPO_ENTREGA_MIN": rng.integers(15, 60, n),
        "KILOMETROS_KM": np.round(rng.uniform(2.0, 12.0, n), 2),
        "CLIMA": ["SOLEADO", "LLUVIOSO", "NUBLADO", "SOLEADO"] * 3,
        "PEDIDO_ID": range(5001, 5001 + n),
    })


def con_nulos() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    n = 20
    df = pd.DataFrame({
        "VALOR": rng.uniform(10, 100, n).round(2),
        "CATEGORIA": ["A", "B", "C", "D"] * 5,
        "NOTA": rng.integers(1, 6, n),
    })
    df.loc[df.index[:3], "VALOR"] = np.nan          # 15% nulos
    df.loc[df.index[::4], "CATEGORIA"] = np.nan     # 50% en una columna → warning >40%
    df["NOTA"] = df["NOTA"].astype("object")        # FIX: upcast explícito (sin FutureWarning)
    df.loc[7, "NOTA"] = "nan"                       # string 'nan' (caso limpieza)
    df = pd.concat([df, df.iloc[[0, 1, 5]]], ignore_index=True)  # 3 dups = ~13%
    return df


def vacio_casi() -> pd.DataFrame:
    return pd.DataFrame({"SOLO": [1, 2, 3]})  # 3 filas → bloqueante


FIXTURES = {
    "limpio_min.csv": limpio_min,
    "mayusculas.csv": mayusculas,
    "con_nulos.csv": con_nulos,
    "vacio_casi.csv": vacio_casi,
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for nombre, fn in FIXTURES.items():
        ruta = os.path.join(OUT, nombre)
        fn().to_csv(ruta, index=False)
        print(f"✅ {ruta}")
