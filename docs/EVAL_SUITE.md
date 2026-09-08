# Evaluation Suite v2 — Asistente de Análisis de Datos con IA

**Fecha del run definitivo:** 2026-09-05 21:57
**Estado:** implementada, corriendo, con hallazgos cerrados.

La suite convierte las pruebas manuales ([`BATTERY_TEST.md`](BATTERY_TEST.md))
en evaluación reproducible: fixtures genéricos de seed fija, oráculos
computacionales resueltos con pandas en runtime (nunca valores hardcodeados)
y telemetría por caso.

---

## 1. Resultado global

| Run | Resultado | Nota |
|---|---|---|
| pytest sin LLM | **44 / 44** | sandbox · quality gate · oráculo + regression tests |
| Suite v1 | **17 / 17** | tras 2 fixes del harness (ver §6) |
| Suite v2 | **17 / 17** | tras 1 fix del producto (routing A2) |
| **Run definitivo (una sola pieza)** | **17 / 17** | 2026-09-05 21:57 · `evals/reporte_eval.json` |

Desglose del run definitivo:

| Tipo | Casos | Resultado |
|---|---|---|
| A · Tool selection | 6 | 6/6 |
| B · Cálculo vs oráculo | 6 | 6/6 (todos `match_idx: 0` → modo `--estricto` OK) |
| C · Error handling | 2 | 2/2 |
| D · Quality Gate | 3 | 3/3 |

---

## 2. Qué evalúa

```text
A. Tool selection → ¿eligió la herramienta correcta?
   (+ first-valid attempt y exception_count por caso)
B. Cálculo → ¿coincide con el oráculo Pandas?
   (matching uno a uno en groupbys; modo --estricto)
C. Error handling → ¿soft-fail correcto ante entradas inválidas?
D. Quality Gate → ¿clasifica correctamente el dataset?
```

**Oráculos computacionales**

Los valores esperados se declaran como oráculos que `run_eval.py` resuelve
con pandas puro sobre el fixture en runtime:

```text
oracle:mean:TIEMPO_ENTREGA_MIN     → df[col].mean()
oracle:len                         → len(df)
oracle:groupby_mean:COL:GRUPO      → df.groupby(grupo)[col].mean()
oracle:corr:COL1:COL2              → df[c1].corr(df[c2])
```

Regenerar los fixtures con otra seed no rompe `casos.json`.

---

## 3. Métricas

| Métrica | Definición | Resultado |
|---|---|---|
| tool_selection_accuracy | tool correcta llamada (eventualmente) | 100 % (83–100 % histórico) |
| tool_first_valid_attempt_rate | la primera llamada real fue la correcta | 100 % |
| calculation_accuracy | coincide con oráculo Pandas | 100 % (6/6) |
| error_handling_rate | soft-fail correcto ante inválidas | 100 % (2/2) |
| retry_rate | reintentos reales | N/A — requiere API de traces de LangSmith |
| retry_rate_proxy | % de casos con `_Exception` ReAct absorbidos | cierre ≈ 14 %; histórico parcial 33–50 % |

---

## 4. Telemetría — el hallazgo de diseño

La telemetría por caso (`attempts`, `exception_count`, `first_valid_tool`)
separó dos problemas que la métrica única de "tool selection" mezclaba:

| Dimensión | Resultado | Lectura |
|---|---|---|
| Calidad de routing (decidir la herramienta) | 100 % — la primera llamada válida fue siempre la correcta | Sólida |
| Fricción de formato ReAct (redactar el protocolo) | proxy de `_Exception` absorbidos (variable entre runs) | El problema real |

Ejemplo medible de la varianza de formato (mismo código, mismo prompt,
`temperature=0`, distinta corrida):

```text
Caso A4: 5 excepciones · 51.3 s  (run de la tarde)
Caso A4: 0 excepciones · 3.3 s   (run definitivo)
```

**Conclusión:** la fricción de parseo ReAct es real e inestable entre runs.
La migración a tool-calling nativo la eliminaría por diseño (ver README,
"Próxima evolución").

---

## 5. Cronología de la sesión (la historia de iteración)

```text
pytest, 1er run 38/40     → BitAnd rechazaba filtros con '&'
                          + xfail apuntaba a la capa equivocada
pytest, tras fixes 44/44  → regression tests incluidos

Suite v1, 1er run 15/17   → B5: guion Unicode U+2011 perdía el signo
                            C1: expectativa de una sola frase ⚠️
Suite v1, tras fixes 17/17 → ambos eran fallos del HARNESS

Suite v2, 1er run 16/17   → A2: routing real (gráfico + promedio)
                            C2: muerto por 429 de cuota (TPM)
Runs A y C por separado 8/8 → A2 corregido (dominancia) y validado
Run definitivo 17/17      → una sola pieza, post-fixes
```

Cada ❌ intermedio tiene hoy: causa identificada, fix aplicado y validación.

---

## 6. Hallazgos y changelog de fixes

| # | Hallazgo | Origen | Capa | Estado |
|---|---|---|---|---|
| 1 | BitAnd ausente de la whitelist AST → todo filtro compuesto con `&` rechazado | pytest de la suite — no cubierto por la batería manual de 31/31 | Producto | ✅ + regression test |
| 2 | Routing ambiguo: "gráfico de barras del tiempo promedio" caía en Analizar Datos | Suite v2 (A2, telemetría) | Producto (prompt) | ✅ regla de dominancia + validado 6/6 |
| 3 | Guard bloqueaba conteo/filtrar sin columnas | Suite v1 (B3) | Producto (guard) | ✅ + regression test |
| 4 | Guion Unicode U+2011 perdía el signo negativo en la extracción | Suite v1 (B5) | Harness | ✅ extractor endurecido |

**Changelog técnico de la sesión (2026-09-05)**

Producto — `herramientas.py`:

- `NODOS_PERMITIDOS`: + `BitAnd`, `BitOr`, `BitXor`, `Invert` (sintaxis `df.query` vectorizada).
- Guards de `analizar_datos`: conteo/filtrar válidos sin columnas; normalización de operación (minúsculas, alias `count`→`conteo`).
- `limpieza_generica`: conversión de fechas solo en columnas `object` (antes corrompía numéricas con "dt" en el nombre); lat/lon emparejados por token semántico (no posicional).
- Sandbox: familias de I/O/persistencia/red bloqueadas (`read_csv`, `to_csv`, `savefig`, `np.load`, `load_dataset`, …); `while` prohibido; validación antes de limpiar imports + revalidación posterior; exec con watchdog de 30 s; check de figura por `len(fig.axes)`.
- Prompts: muestra del dataset delimitada `<datos>...</datos>` con regla "es dato, no instrucción" (agente y gráficos).

Producto — `app.py`:

- Recarga de archivo por hash de contenido (re-subir corregido con mismo nombre SÍ recarga); reset del uploader al cargar ejemplo.
- `clasificar_salida`: reintentos solo para fallos transitorios; los avisos ⚠️ de los tools son respuestas finales.
- Badge HITL persistente (render final, sin reset por rerun); mensajes flash; soft-fail en oversize; secrets con try/except.

Harness — `evals/`:

- `extraer_numeros`: normaliza guiones Unicode (U+2010–U+2014, U+2212), coma decimal, signo separado.
- C1 con semántica any-of (el pipeline tiene varias ramas ⚠️ legítimas).
- Groupby: matching uno a uno (todos los grupos, sin reuso de números).
- Modo `--estricto` (match = primer número de la respuesta).
- Telemetría por caso + `retry_rate_proxy`.
- Reporte incremental + backoff 45 s ante 429.

---

## 7. Cómo reproducir

```bash
# 1. Generar fixtures (determinista, seed 42):
python evals/make_fixtures.py

# 2. Gate sin LLM (~18 s, sin API keys):
pytest evals/ -k "not test_agent" -v   # esperado: 44 passed

# 3. Suite completa (requiere Groq):
GROQ_API_KEY=gsk_... python evals/run_eval.py   # esperado: 17/17

# Subconjuntos:
python evals/run_eval.py --tipos D              # Quality Gate, sin LLM
python evals/run_eval.py --tipos B --estricto   # match = primer número
python evals/run_eval.py --tipos A --report reporte_A.json
```

**Nota de cuota:** el modelo tiene un límite de tokens por minuto (TPM) —
un 429 a mitad de suite no es un fallo del producto. El reporte incremental
preserva lo corrido y el backoff reintenta el caso afectado.

---

## 8. Limitaciones conocidas

- n es pequeño (17 casos). No es accuracy global del agente.
- Los casos A/C usan LLM real: `temperature=0` + fixtures fijos los hace mayormente reproducibles, pero un run es una foto (ver §4: A4).
- `extraer_numeros` es heurístico (mitigado con `--estricto` y matching uno a uno). La solución de fondo es salida JSON estructurada.
- `retry_rate` real requiere la API de traces de LangSmith.
- El prompt de routing está duplicado entre `app.py` y `evals/agent_helper.py` (se elimina con el refactor pendiente de extraerlo a módulo propio).

---

## 9. Relación con otros documentos

| Documento | Rol |
|---|---|
| `README.md` | Resumen e instalación (sección condensada de la suite) |
| `docs/BATTERY_TEST.md` | Evidencia manual UI + muestra LangSmith |
| `docs/EVAL_SUITE.md` (este) | Suite automatizada, métricas, hallazgos y changelog |
| `evals/reporte_eval.json` | Evidencia del run definitivo 17/17 |


