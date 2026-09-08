# Batería de pruebas manual — Asistente de Análisis de Datos con IA

**Fecha de cierre de fase:** 2026-09-05  
**Versión / commit probado:** `<hash-del-commit>`  
**Entorno:** Google Colab y Streamlit  
**Python:** 3.13 (Colab) / según entorno local  

**Criterio de aprobación:** ≥ 80 % en cálculos (**C**) + grupos/filtros/correlación (**G**) + visualizaciones (**V**), con **C4 (conteo total de filas) obligatorio**, y errores controlados sin tumbar la app.

| Métrica | Resultado |
|---------|-----------|
| Batería funcional manual | **31 / 31 (100 %) — APROBADO** |
| Tool selection (muestra manual LangSmith) | **6 / 6 casos en la muestra** |

> **Nota metodológica:** la selección de herramientas aquí es una **muestra manual de seis trazas**, no una accuracy global del agente.  
> Para medición reproducible con fixtures, oráculos Pandas y telemetría, ver [`EVAL_SUITE.md`](EVAL_SUITE.md).

> **Convenciones:** **C** = cálculos; **G** = grupos, filtros y correlación; **V** = visualizaciones/gráficos.  
> Los valores con `~` o `≈` se muestran redondeados; los resultados se contrastaron contra Pandas sobre el DataFrame cargado.

> **Quality Gate:** el score (p. ej. ~91/100) corresponde a la **fórmula interna** de esta aplicación; no es una certificación externa de calidad del dataset.

---

## 1. Entorno

| Elemento | Valor |
|----------|--------|
| UI | Streamlit |
| Agente | LangChain ReAct (`AgentExecutor`) |
| LLM | Groq `openai/gpt-oss-120b` |
| Trazas | LangSmith |
| Dataset A | `datos_entregas.csv` — ~43 739 filas × 16 columnas |
| Dataset B | Mini CSV (~10 filas), columnas en MAYÚSCULAS |

---

## 2. Dataset A — Entregas (resumen)

| Bloque | Resultado | Notas |
|--------|-----------|--------|
| Quality gate / HITL / limpieza | ✅ | Score interno ~91/100; advertencias (p. ej. `categoria_producto`) |
| Auditoría directa (info + stats) | ✅ | Sin pasar por ReAct (ruta de UI a propósito) |
| **C1** promedio `tiempo_entrega` | ✅ | ~124.91 |
| **C2** mínimo experiencia colaborador | ✅ | 1 |
| **C3** máximo `tiempo_entrega` | ✅ | 270 |
| **C4** total de pedidos (**obligatorio**) | ✅ | **43 739** |
| **G** grupos / filtro Soleado / correlación | ✅ | Soleado ~103.66; r ≈ **−0.252** |
| Unidades en texto | ✅ | Sin inventar unidad si el nombre no la indica |
| Columna inexistente | ✅ | Soft-fail; app continúa |
| **V** gráficos V1–V5 | ✅ | Barras, boxplot, heatmap, histograma, dispersión |
| Export CSV + ZIP | ✅ | |

**Subtotal A: 23 / 23**

---

## 3. Dataset B — Mini MAYÚSCULAS

| Caso | Resultado |
|------|-----------|
| Auditoría con columnas en MAYÚSCULAS | ✅ |
| Pregunta en minúsculas → columna real (p. ej. `TIEMPO_ENTREGA`) | ✅ |
| Total de pedidos / filas del mini set | ✅ |
| Promedio por clima | ✅ |
| Unidades por nombre (`_MIN`, `_KM`) cuando aplica | ✅ |
| Gráficos (barras, dispersión) | ✅ |

**Subtotal B: 8 / 8**

---

## 4. Tool selection — muestra manual (LangSmith)

Casos agénticos revisados en trazas. La auditoría por **botones de UI** no se incluye (ruta directa a propósito; no se mezcla con métricas del agente).

| # | Pregunta | Tool esperada | Tool observada |
|---|----------|---------------|----------------|
| 1 | ¿Promedio de tiempo_entrega? | Analizar Datos | Analizar Datos |
| 2 | Gráfico de barras del tiempo medio por clima | Generar Gráfico | Generar Gráfico |
| 3 | Correlación experiencia y tiempo | Analizar Datos | Analizar Datos |
| 4 | Heatmap clima × vehículo | Generar Gráfico | Generar Gráfico |
| 5 | Promedio de columna_que_no_existe | Analizar Datos | Analizar Datos |
| 6 | columna_que_no_existe | Analizar Datos | Analizar Datos |

**Resultado de la muestra: 6 / 6.**

En 5–6 la salida es un aviso de columna/operación no válida y el listado de columnas (soft-fail), no un reporte de información general.

### Nota metodológica

Esta tabla es una **muestra manual** de enrutamiento.  
No afirma que el agente tenga “100 % de tool selection accuracy” en todos los inputs posibles.  
La [Evaluation Suite](EVAL_SUITE.md) mide tool selection y cálculo de forma reproducible sobre fixtures genéricos (17 casos A/B/C/D + telemetría first-valid / proxy de `_Exception`).

---

## 5. Batería corta del código parcheado (post-hardening)

Validación adicional tras hash de carga, HITL, soft-fail de reintentos y sandbox AST:

| Bloque | Resultado |
|--------|-----------|
| Unitario `es_codigo_seguro` (os, read_csv, to_csv, savefig, while, plot OK) | ✅ |
| Timeout `_exec_restringido` (~3 s) | ✅ |
| Columna inexistente sin 3 reintentos inútiles | ✅ |
| Ejemplo + limpieza + badge HITL | ✅ |
| Gráfico generado | ✅ |
| Conteo total tras fix `columnas []` + operación `conteo` | ✅ **43 739** |

---

## 6. Criterio ≥ 80 %

| Condición | ¿Cumple? |
|-----------|----------|
| ≥ 80 % en **C** + **G** + **V** | Sí (100 %) |
| **C4** obligatorio | Sí |
| Errores controlados sin tumbar la app | Sí |
| Muestra tool selection ≥ 80 % | Sí (6/6 en la muestra) |

**Veredicto de esta fase: APROBADO.**

---

## 7. Limitaciones de la batería manual

1. No cubrió filtros compuestos con `&` (detectado después por la suite / AST).  
2. No midió telemetría first-valid vs `_Exception` de parse ReAct.  
3. Depende del dataset cargado y de la sesión Streamlit (túnel/Colab puede perder estado).  
4. No sustituye `pytest evals/` ni `python evals/run_eval.py`.

---

## 8. Relación con otros documentos

| Documento | Rol |
|-----------|-----|
| `README.md` | Resumen e instalación |
| `docs/BATTERY_TEST.md` (este) | Evidencia manual UI + muestra LangSmith |
| `docs/EVAL_SUITE.md` | Suite automatizada, métricas y hallazgos |

---

## 9. Cómo reproducir (manual)

1. Configurar `GROQ_API_KEY` y `LANGCHAIN_API_KEY`.  
2. `streamlit run app.py`.  
3. Cargar CSV o dataset de ejemplo.  
4. Repetir casos de las secciones 2–5 y contrastar números con pandas / trazas LangSmith.  
5. Tras el push, sustituir `<hash-del-commit>` en la cabecera por el commit que cerró esta evidencia.
