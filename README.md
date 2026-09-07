# 🦜 Asistente de Análisis de Datos con IA

Asistente conversacional híbrido para explorar, validar, limpiar, consultar y visualizar datasets CSV tabulares genéricos.

Combina un agente ReAct basado en LangChain y Groq con cálculo determinista mediante Pandas, Quality Gate, Human-in-the-Loop (HITL), ejecución restringida de código para gráficos y trazabilidad mediante LangSmith.

**Principio de diseño:** el LLM no tiene que hacerlo todo. Cada responsabilidad se delega al componente más adecuado.

![tests](https://img.shields.io/badge/tests-passing-brightgreen) ![Python](https://img.shields.io/badge/Python-3.x-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red) ![Pandas](https://img.shields.io/badge/Pandas-data-150458) ![Groq](https://img.shields.io/badge/Groq-LLM-orange) ![LangSmith](https://img.shields.io/badge/LangSmith-observability-purple) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 Contenido

[Aplicación desplegada](#-aplicación-desplegada) · [¿Qué es y qué no es?](#-qué-es-y-qué-no-es) · [Objetivo](#-objetivo)
[Características](#-características-principales) · [Quality Gate](#️-quality-gate) · [Human-in-the-Loop](#-human-in-the-loop)
[Pandas como verdad](#-pandas-como-fuente-de-verdad) · [Ejecución restringida](#-ejecución-restringida-de-gráficos) · [Auditoría directa](#-auditoría-directa)
[Enrutamiento del agente](#-enrutamiento-del-agente) · [Política de unidades](#-política-de-unidades) · [Soft-fail y reintentos](#-soft-fail-reintentos-y-backoff)
[Observabilidad](#-observabilidad-con-langsmith) · [Exportación](#-exportación) · [Módulos](#-módulos-de-la-aplicación)
[Arquitectura](#️-arquitectura) · [Responsabilidades](#-responsabilidades-por-componente) · [Ejemplo de conversación](#-ejemplo-de-conversación)
[Pruebas y validación](#-pruebas-y-validación) · [Evaluation Suite v2](#-evaluation-suite-v2) · [Próxima evolución](#-próxima-evolución-de-react-a-tool-calling-nativo)
[Capturas](#-capturas) · [Configuración](#️-configuración) · [Instalación local](#-instalación-local)
[Deployment](#-deployment-en-streamlit-community-cloud) · [Estructura del repo](#-estructura-del-repositorio) · [Seguridad](#-seguridad-y-privacidad)
[Limitaciones](#️-limitaciones-conocidas) · [Aprendizajes](#-aprendizajes-clave) · [Estado del proyecto](#estado-del-proyecto)

---

## 🚀 Aplicación desplegada

La aplicación está publicada en Streamlit Community Cloud.

🔗 **Demo:** https://<tu-app>.streamlit.app/ ← *pendiente de deploy*

Durante el desarrollo y las pruebas se utiliza Google Colab. El túnel se utiliza únicamente para exponer temporalmente la aplicación durante la experimentación; no forma parte del deployment final.

---

## 🧠 ¿Qué es y qué no es?

| ✅ Sí | ❌ No |
|---|---|
| Sistema analítico conversacional híbrido | Agente autónomo con planificación libre |
| Enrutamiento de consultas mediante ReAct | LLM utilizado como fuente de verdad numérica |
| Cálculo determinista con Pandas | Cálculos numéricos delegados al modelo |
| Herramientas especializadas según la tarea | Un único flujo que hace todo mediante LLM |
| Quality Gate y supervisión HITL | Aceptación ciega de entradas o salidas |
| Validación AST y ejecución restringida | Ejecución arbitraria de código |
| Evaluación automatizada con oráculos | Pruebas solo manuales sobre un único dataset |
| Trazabilidad mediante LangSmith | Procesos sin observabilidad |

**En una frase**
El LLM interpreta y enruta, Pandas calcula, los componentes especializados ejecutan y LangSmith permite rastrear lo ocurrido.

---

## 🎯 Objetivo

El proyecto busca resolver un problema común: permitir que un usuario trabaje con un CSV mediante lenguaje natural sin convertir al LLM en responsable de los cálculos ni de todas las decisiones del sistema.

La arquitectura separa:

```
Interpretación
      ↓
Enrutamiento
      ↓
Herramienta especializada
      ↓
Cálculo / ejecución controlada
      ↓
Resultado
      ↓
Presentación
      ↓
Trazabilidad
```

---

## ✨ Características principales

| Área | Implementación |
|---|---|
| Quality Gate | Score 0–100, problemas bloqueantes y advertencias |
| HITL | Estados explícitos para supervisión y manejo de excepciones |
| Limpieza | Duplicados, nan → NaN, coordenadas (0,0), fechas e imputación conservadora |
| Auditoría | Información general y estadísticas ejecutadas directamente |
| Análisis | Promedios, sumas, conteos, máximos, mínimos, correlaciones, outliers, filtros y agrupaciones |
| Cálculo | Pandas como fuente de verdad numérica |
| Case‑insensitive | Resolución de preguntas independientemente de mayúsculas/minúsculas |
| Unidades | No se inventan unidades cuando no existe evidencia |
| Gráficos | LLM → código Python → AST → ejecución restringida con watchdog |
| Sandbox | I/O, persistencia y red bloqueadas; `while` prohibido; built‑ins mínimos |
| Soft‑fail | Errores de operación tratados sin derribar la sesión |
| Reintentos | Solo fallos transitorios (máx. 3, backoff ante rate limits) |
| Observabilidad | LangSmith obligatorio para trazabilidad |
| Evaluación | Suite automatizada con oráculos Pandas (ver [Evaluation Suite v2](#-evaluation-suite-v2)) |
| Exportación | CSV limpio, bitácora, reportes y gráficos en ZIP |

---

## 🛡️ Quality Gate

Antes de procesar el dataset se realiza una validación de calidad.

Se consideran, entre otros:

- filas y columnas mínimas;
- porcentaje global de nulos;
- duplicados;
- columnas completamente vacías;
- columnas con alta proporción de nulos.

El resultado produce un score de 0 a 100 y distingue entre:

```
Dataset bloqueante
      ↓
     STOP
      ↓
     HITL
      ↓
Usuario corrige o sustituye el archivo
```

y:

```
Advertencias
      ↓
La aplicación continúa
      ↓
     HITL
      ↓
El usuario decide
```

Después de la limpieza se ejecuta nuevamente el Quality Gate para obtener un score post‑limpieza.

---

## 👤 Human‑in‑the‑Loop

La aplicación utiliza estados explícitos:

```
READY → QUALITY_WARNING → QUALITY_BLOCKED → EXECUTING → RETRYING → FAILED → SUCCESS
```

El principio es:

> La máquina ejecuta y avisa; el humano decide cuando corresponde.

El HITL no pretende convertir cada operación en una aprobación manual. Se utiliza principalmente para gestionar bloqueos, advertencias y fallos donde continuar automáticamente no es apropiado.

---

## 🔢 Pandas como fuente de verdad

El LLM no realiza los cálculos numéricos finales.

El flujo de análisis es:

```
Pregunta del usuario
        ↓
LLM interpreta intención
        ↓
Plan estructurado (JSON)
        ↓
Pandas ejecuta
        ↓
Resultado determinista
        ↓
LLM presenta el resultado
```

Ejemplos de operaciones:

- promedio;
- suma;
- conteo;
- máximo;
- mínimo;
- correlación;
- detección de outliers;
- descripción estadística;
- filtros;
- agrupaciones.

Esta separación reduce el riesgo de que una respuesta lingüística sustituya al cálculo real.

---

## 🔐 Ejecución restringida de gráficos

Los gráficos utilizan un flujo diferente porque el LLM genera código Python.

```
Solicitud del usuario
        ↓
LLM genera código
        ↓
Limpieza de salida
        ↓
Validación AST (antes de limpiar imports)
        ↓
Bloqueo de módulos / funciones / familias de I/O
        ↓
Revalidación post-limpieza
        ↓
Built‑ins mínimos
        ↓
UN solo exec restringido con watchdog (30 s)
        ↓
Figura (validada: debe tener ejes)
        ↓
PNG
```

El sistema bloquea, entre otros:

- **ejecución dinámica:** `eval`, `exec`, `compile`, `__import__`;
- **sistema y red:** `os`, `sys`, `subprocess`, `socket`, `requests`, `urllib`;
- **lectura de archivos y URLs:** `read_csv`, `read_pickle`, `read_excel`, `read_json`, `read_sql`, …;
- **escritura y persistencia:** `to_csv`, `to_pickle`, `to_excel`, `savefig`, `np.save`, `dump`, …;
- **bucles `while`** (riesgo de bucle infinito);
- **atributos dunder** fuera de una lista mínima.

Además, la muestra del dataset que se pasa al prompt va delimitada (`<datos>...</datos>`) con una regla explícita: es **dato, no instrucción** — mitigación de inyección de prompt desde el contenido del CSV.

**Nota de seguridad**

La implementación debe describirse como:

> Validación AST + ejecución restringida con built‑ins mínimos + watchdog de tiempo.

No se presenta como un sandbox de aislamiento absoluto. La validación reduce la superficie de riesgo, pero no sustituye un aislamiento de proceso o contenedor cuando ese nivel de seguridad sea un requisito. Los vectores de exfiltración conocidos (p. ej. `pd.read_csv('.streamlit/secrets.toml')` y derivados) están cubiertos por tests automatizados.

---

## 📊 Auditoría directa

Las funciones de auditoría se ejecutan directamente y no dependen del agente ReAct.

**Información general**

Incluye:

- tipos de datos;
- valores nulos;
- duplicados;
- completitud;
- estado de cada variable.

**Estadísticas**

Incluye:

- mínimo;
- media;
- mediana;
- máximo;
- desviación estándar;
- outliers mediante IQR.

Esto reduce la dependencia del LLM en operaciones básicas de diagnóstico.

---

## 🤖 Enrutamiento del agente

El agente ReAct dispone de herramientas especializadas:

- Información DF
- Resumen Estadístico
- Analizar Datos
- Generar Gráfico
- Limpiar Datos

Ejemplo de política de routing:

```
información / nulos / duplicados
        → Información DF

estadísticas globales
        → Resumen Estadístico

promedio / suma / conteo / correlación / outliers
        → Analizar Datos

gráfico / barras / heatmap / visualización
        → Generar Gráfico SIEMPRE (la visualización
          DOMINA sobre el cálculo, aunque la pregunta
          también mencione promedios o estadísticas)

limpieza
        → Limpiar Datos
```

La regla de dominancia fue añadida tras un hallazgo de la Evaluation Suite: "gráfico de barras del tiempo promedio" contenía keywords de dos reglas y el agente caía hacia el cálculo (ver [Hallazgos](#hallazgos-producidos-por-la-suite)).

La auditoría mediante botones de la interfaz sigue una ruta directa y, deliberadamente, no pasa por ReAct.

---

## 📏 Política de unidades

La aplicación evita asumir unidades que el dataset no especifica.

```
Nombre de columna: TIEMPO_ENTREGA
        ↓
sin unidad explícita
        ↓
no inventar "minutos"

Nombre de columna: TIEMPO_ENTREGA_MIN
        ↓
evidencia suficiente
        ↓
puede utilizar "min"
```

También se respeta una unidad indicada explícitamente por el usuario.

El principio es:

> Sin evidencia, no se inventa la unidad.

---

## 🔁 Soft‑fail, reintentos y backoff

Los fallos parciales no deberían convertir una operación fallida en una sesión inutilizable.

Los reintentos se aplican solo a fallos transitorios del orquestador (excepciones, rate limits, salidas vacías o límites de iteración):

```
Intento 1
   ↓ fallo transitorio
Intento 2 (backoff si es 429)
   ↓
Intento 3
   ↓
FAILED
```

Los avisos con formato que producen los tools (⚠️ columna no encontrada…) son respuestas finales legítimas: el pipeline funcionó como fue diseñado y el usuario recibió información accionable. Reintentarlos con `temperature=0` reproduciría el mismo resultado tres veces.

Después de los reintentos agotados:

```
FAILED
   ↓
La aplicación permanece disponible
   ↓
     HITL
   ↓
Usuario decide el siguiente paso
```

---

## 🔎 Observabilidad con LangSmith

LangSmith es obligatorio para la aplicación.

La trazabilidad permite revisar:

- ejecución del AgentExecutor;
- herramienta seleccionada;
- llamadas al modelo;
- entradas y salidas;
- errores;
- reintentos;
- tiempos;
- metadatos de ejecución.

La interfaz mantiene una presentación limpia y el detalle técnico queda disponible en LangSmith.

**Principio:** la UI muestra el resultado; LangSmith permite investigar cómo se obtuvo.

---

## 📦 Exportación

La aplicación permite descargar:

- CSV limpio
- Bitácora de limpieza
- Reporte de información
- Reporte estadístico
- Gráficos

como archivos individuales o dentro de un ZIP.

---

## 🧩 Módulos de la aplicación

| Pestaña | Función |
|---|---|
| 📁 Datos | Exploración, limpieza, validación post‑limpieza y bitácora |
| 🔍 Auditoría | Integridad, estadísticas y outliers |
| 🔎 Análisis | Consultas en lenguaje natural mediante ReAct |
| 📊 Gráficos | Visualizaciones generadas a partir de lenguaje natural |
| 📚 Historial | Preguntas y respuestas de la sesión |

---

## 🏗️ Arquitectura

```
                         Usuario
                            │
                            ▼
                    ┌───────────────┐
                    │   Streamlit   │
                    │      UI       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Quality Gate  │
                    └───────┬───────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
            Bloqueante            OK / Warning
                 │                     │
                 ▼                     ▼
               STOP               Agente ReAct
                 │                     │
                HITL          ┌────────┼─────────┐
                               │        │         │
                               ▼        ▼         ▼
                           Analizar  Gráfico   Limpiar
                             Datos               Datos
                               │        │         │
                               ▼        ▼         ▼
                            Pandas   LLM →      Limpieza
                            cálculo  código     con reglas
                           determinista │        validadas
                                      AST
                                        │
                                 exec restringido
                                 (watchdog 30 s)
                                        │
                                        ▼
                                    Resultado
                                        │
                                        ▼
                                      HITL
                                        │
                                        ▼
                              Presentación al usuario

        LangSmith observa transversalmente todo el flujo
        (tools elegidas, llamadas al modelo, errores, tiempos):
        no es un paso final, es la capa de observabilidad.
```

---

## 🧱 Responsabilidades por componente

| Componente | Responsabilidad |
|---|---|
| LLM / Groq | Interpretación, routing, generación de código de gráficos y presentación |
| ReAct | Enrutamiento hacia la herramienta adecuada |
| Pandas | Cálculo determinista |
| AST | Validación del código y de los filtros generados |
| Quality Gate | Validación de entrada |
| HITL | Supervisión y decisión humana ante excepciones |
| Watchdog | Límite de tiempo en la ejecución restringida |
| Streamlit | Interfaz y experiencia de usuario |
| LangSmith | Observabilidad y trazabilidad |
| Evaluation Suite | Verificación automatizada del comportamiento del sistema |

---

## 💬 Ejemplo de conversación

**Usuario**

> Promedio de TIEMPO_ENTREGA_MIN

**Asistente**

> El promedio de TIEMPO_ENTREGA_MIN es 37.98 min.

**Usuario**

> Promedio de TIEMPO_ENTREGA

**Asistente**

> El promedio de TIEMPO_ENTREGA es 37.98. La unidad no está especificada en el nombre de la columna.

**Usuario**

> Genera un gráfico de barras del tiempo promedio por clima.

**Asistente**

> Se genera el gráfico solicitado y se ofrece la descarga en formato PNG.

**Usuario**

> Promedio de columna_que_no_existe

**Asistente**

> ⚠️ No se indicó una columna válida para el cálculo. Columnas disponibles: [...]

La aplicación continúa disponible.

**Usuario** *(sube un CSV con advertencias de calidad)*

**Asistente**

> 🟡 Dataset usable. Score: 78/100 — Duplicados moderados (13.0%).
> ⚠️ Advertencias (no bloquean; tú decides): [...]

El usuario decide si continuar.

---

## 🧪 Pruebas y validación

La batería manual completa está documentada en:

```
docs/BATTERY_TEST.md
```

La evaluación automatizada vive en `evals/`, documentada en detalle en `docs/EVAL_SUITE.md`.

**Resultados actuales**

| Métrica | Resultado |
|---|---|
| Batería funcional manual | 31 / 31 — 100 % |
| pytest sin LLM (sandbox · quality gate · oráculo) | 44 / 44 |
| Evaluation Suite v2 (A/B/C/D) | 17 / 17 — 100 % · run completo |
| Tool selection · eventual | 100 % (rango histórico 83–100 %) |
| Tool selection · primera llamada válida | 100 % |
| Cálculo vs oráculo Pandas | 100 % (6/6) |
| Error handling (soft-fail) | 100 % (2/2) |

**Nota de honestidad:** los casos con LLM real corren con `temperature=0` y fixtures de seed fija, lo que los hace mayormente reproducibles, pero un run es una foto. El 83 % del rango histórico corresponde a un bug de routing detectado, corregido y validado (ver [Hallazgos](#hallazgos-producidos-por-la-suite)).

La batería manual cubre:

- Quality Gate;
- HITL;
- limpieza;
- auditoría;
- cálculos;
- agrupaciones;
- filtros;
- correlación;
- manejo de columnas inexistentes;
- case‑insensitive;
- política de unidades;
- gráficos;
- exportación.

**Importante:** el 6/6 histórico de tool selection por muestra manual de LangSmith fue sustituido por la medición reproducible de la suite.

---

## 🧪 Evaluation Suite v2

La suite existe y corre. Convierte las pruebas manuales en evaluación reproducible sobre fixtures genéricos de seed fija, con oráculos computacionales (`oracle:mean:COL`) resueltos con pandas en runtime — nunca valores hardcodeados.

**Estructura**

```
evals/
├── fixtures/               # 4 CSVs deterministas (seed 42)
│   ├── limpio_min.csv      #   → ok
│   ├── mayusculas.csv      #   → ok (case-insensitive)
│   ├── con_nulos.csv       #   → warning
│   └── vacio_casi.csv      #   → bloqueante
├── make_fixtures.py        # regenera los fixtures byte a byte
├── casos.json              # 17 casos: pregunta + expectativa
├── agent_helper.py         # agente para evals (aislado, sin memoria)
├── run_eval.py              # suite + métricas + reporte JSON incremental
├── test_sandbox.py          # AST · filtros · timeout — sin LLM
├── test_quality_gate.py     # clasificación D — sin LLM
├── test_oracle.py           # engine vs oráculo + regression tests — sin LLM
└── test_agent.py            # routing con LLM real (skip sin API key)
```

**Qué evalúa**

```
A. Tool selection   → ¿eligió la herramienta correcta?
                      (+ first-valid attempt y exception_count por caso)
B. Cálculo          → ¿coincide con el oráculo Pandas?
                      (matching uno a uno en groupbys; modo --estricto)
C. Error handling   → ¿soft-fail correcto ante entradas inválidas?
D. Quality Gate     → ¿clasifica correctamente el dataset?
```

**Cómo correrla**

```bash
# CI / local sin LLM — ~18 segundos, sin API keys:
pytest evals/ -k "not test_agent" -v

# Suite completa (con Groq + LangSmith):
GROQ_API_KEY=gsk_... python evals/run_eval.py

# Subconjuntos:
python evals/run_eval.py --tipos D               # solo Quality Gate, sin LLM
python evals/run_eval.py --tipos B --estricto    # match = primer número
python evals/run_eval.py --tipos A --report reporte_A.json
```

**Métricas**

| Métrica | Definición | Resultado |
|---|---|---|
| tool_selection_accuracy | tool correcta llamada (eventualmente) | 100 % (83–100 % histórico) |
| tool_first_valid_attempt_rate | la primera llamada real fue la correcta | 100 % |
| calculation_accuracy | coincide con oráculo Pandas | 100 % (6/6) |
| error_handling_rate | soft-fail correcto ante inválidas | 100 % (2/2) |
| retry_rate | reintentos reales | N/A — requiere API de traces de LangSmith |
| retry_rate_proxy | % de casos con `_Exception` ReAct absorbidos | 33–50 % |

El proxy de retry revela que la fricción restante es de formato ReAct (parseo), no de routing: en los runs con excepciones, la primera llamada válida fue siempre la herramienta correcta. La migración a tool-calling nativo eliminaría esa fricción por diseño.

**Hallazgos producidos por la suite**

| Hallazgo | Origen | Estado |
|---|---|---|
| Filtros compuestos con `&` rechazados (nodo BitAnd ausente de la whitelist AST) | Primer run — no cubierto por la batería manual de 31/31 | ✅ Corregido + regression test |
| Ambigüedad de routing: "gráfico de barras del tiempo promedio" → caía en Analizar Datos | Run de tool selection (A2) | ✅ Corregido (regla de dominancia) + validado |
| Parser sin columnas bloqueaba conteo/filtrado global | Caso B3 | ✅ Corregido + regression test |
| Guiones Unicode (U+2011) perdían el signo negativo en la evaluación de respuestas | Caso B5 — fallo del harness, no del producto | ✅ Corregido en el extractor |

La batería manual validó que lo construido funciona; la suite encontró lo que la batería no cubría. Esa es la diferencia, y la razón de su existencia.

**Limitaciones conocidas de la suite**

- n es pequeño (17 casos). No es accuracy global del agente.
- Los casos A/C usan LLM real: estables pero un run es una foto.
- `extraer_numeros` es heurístico (mitigado con `--estricto` y matching uno a uno).
- `retry_rate` real requiere la API de traces de LangSmith.

---

## 🔮 Próxima evolución: de ReAct a tool-calling nativo

La telemetría de la suite identificó con precisión dónde está la fricción restante del agente:

| Dimensión | Resultado | Lectura |
|---|---|---|
| Calidad de routing (decidir la herramienta) | 100 % — la primera llamada válida fue siempre la correcta | Sólida |
| Fricción de formato ReAct (redactar el protocolo) | 33–50 % de casos con `_Exception` absorbidos | El problema |

ReAct clásico exige al modelo dos tareas: decidir la herramienta y escribir el protocolo (`Thought:` / `Action:`) sin errores de formato. El agente decide bien; falla al redactar. Tool-calling nativo elimina la segunda tarea de raíz: el modelo devuelve llamadas estructuradas en vez de texto parseable.

**Beneficios esperados**

```
· retry_rate_proxy:        33–50 %  →  ~0 %
· Latencia en casos con excepciones en cadena
                           ~51 s     →  ~solo la llamada a la tool
· Tokens por consulta:     menores (sin reintentos de parseo)
· Superficie de error:     sin parser propio que mantener
```

**Plan**

```
1. Benchmark controlado:    10 preguntas → ReAct vs tool-calling
                             (excepciones, latencia, tokens)
2. Migración del agente:    mismas tools, mismas reglas de routing,
                             nuevo andamiaje — el prompt pierde el bloque
                             de formato y deja de estar duplicado
3. Validación:              la MISMA suite, sin cambios —
                             pytest 44/44 debe mantenerse,
                             A/C debe mantener 100 % de tool selection
```

La migración se valida con la misma suite que motivó el cambio: si tool selection se mantiene en 100 % y el proxy de excepciones cae, el cambio está justificado por datos propios, no por moda de stack.

---

## 📸 Capturas

Pendiente incorporar capturas reales del proyecto.

Se propone:

```
docs/images/
├── datos.png
├── auditoria.png
├── analisis.png
└── graficos.png
```

---

## ⚙️ Configuración

**Variables obligatorias**

```
GROQ_API_KEY=tu_clave_groq
LANGCHAIN_API_KEY=tu_clave_langsmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Asistente-Colab
```

También pueden configurarse mediante:

```
.streamlit/secrets.toml
```

No subir este archivo al repositorio.

**Límites de la aplicación**

| Límite | Valor | Dónde |
|---|---|---|
| Tamaño máximo de CSV | 15 MB | `MAX_FILE_MB` en `app.py` |
| Timeout del código de gráficos | 30 s | `EXEC_TIMEOUT_S` en `herramientas.py` |

---

## 💻 Instalación local

```bash
git clone https://github.com/fsoteou-cyber/Llm-data-agent.git
cd Llm-data-agent
python -m venv .venv
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
streamlit run app.py
```

---

## 🚀 Deployment en Streamlit Community Cloud

1. Subir el repositorio a GitHub.
2. Crear una nueva aplicación en Streamlit Community Cloud.
3. Seleccionar el repositorio.
4. Configurar:
   - Main file: `app.py`
5. Añadir los secrets:

```
GROQ_API_KEY = "gsk_..."
LANGCHAIN_API_KEY = "lsv2_..."
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "Asistente-Colab"
```

6. Desplegar.

Cada actualización enviada al repositorio puede activar una nueva versión de la aplicación.

---

## 📁 Estructura del repositorio

```
/
├── app.py
├── herramientas.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── .github/workflows/ci.yml      # CI: pytest sin LLM
│
├── docs/
│   ├── BATTERY_TEST.md           # batería manual
│   ├── EVAL_SUITE.md             # suite v2: métricas, hallazgos, changelog
│   └── reportes/                 # evidencia de runs de la suite
│
├── data/
│   └── datos_entregas.csv        # dataset de ejemplo
│
└── evals/                        # Evaluation Suite v2 (implementada)
    ├── make_fixtures.py
    ├── casos.json
    ├── agent_helper.py
    ├── run_eval.py
    ├── test_sandbox.py
    ├── test_quality_gate.py
    ├── test_oracle.py
    ├── test_agent.py
    ├── fixtures/                 # 4 CSVs regenerables (seed 42)
    └── reporte_eval.json
```

---

## 🔒 Seguridad y privacidad

**Código generado**

- Validación AST antes de la limpieza de imports, con revalidación posterior.
- Bloqueo de módulos peligrosos y de familias completas de I/O, persistencia y red: `read_csv`, `read_pickle`, `read_excel`, `to_csv`, `to_pickle`, `savefig`, `imread`, `np.load`, `dump`, `load_dataset`, entre otras.
- `while` prohibido (riesgo de bucle infinito).
- Built‑ins mínimos.
- Watchdog de 30 s en el exec restringido: un cálculo descontrolado aborta en vez de congelar la sesión (limitación documentada: el hilo no se puede matar; queda abandonado como daemon).
- Delimitado de datos en prompts: la muestra del dataset va entre `<datos>...</datos>` con regla explícita de "es dato, no instrucción" — tanto en el prompt del agente como en el de gráficos.

**Filtros**

Las expresiones para `df.query()` se validan mediante whitelist de nodos AST, incluidos los operadores bitwise (`&`, `|`, `~`) que exige la sintaxis de pandas.

**Credenciales**

Nunca subir:

```
.env
.streamlit/secrets.toml
API keys
tokens
```

**Datos**

Los datasets cargados se mantienen durante la sesión de Streamlit y no forman parte del repositorio. El `.gitignore` excluye todos los CSV por defecto, con excepciones explícitas para los fixtures y el dataset de ejemplo.

---

## ⚠️ Limitaciones conocidas

- El agente utiliza ReAct clásico: el routing es sólido (first-valid 100 % en la suite), pero persiste fricción de formato que se absorbe con reintentos de parseo (ver [Próxima evolución](#-próxima-evolución-de-react-a-tool-calling-nativo)).
- La validación AST reduce la superficie de riesgo, pero no proporciona aislamiento de proceso.
- El watchdog no puede matar el hilo del exec: si el timeout vence, el hilo queda abandonado hasta terminar por su cuenta.
- La suite tiene n pequeño (17 casos) y los casos con LLM son una foto por más reproducibles que sean.
- `extraer_numeros` de la evaluación es heurístico (mitigado con `--estricto` y matching uno a uno).
- Algunas etiquetas generadas en gráficos pueden ser más agresivas con las unidades que la respuesta textual.
- El sistema no debe interpretarse como un agente autónomo con planificación libre o memoria de trabajo a largo plazo.

Estas limitaciones forman parte de la definición real del sistema y se documentan deliberadamente.

---

## 📈 Aprendizajes clave

Este proyecto permitió consolidar varios principios de diseño de sistemas de IA.

**1. Separación de responsabilidades**

El LLM no tiene por qué resolver cada tarea.

```
LLM        → interpretar
Pandas     → calcular
AST        → validar
Herramientas → ejecutar
LangSmith  → observar
Humano     → decidir cuando corresponde
```

**2. Gobernanza antes que autonomía**

No todo lo que un modelo puede generar debe ejecutarse automáticamente.

**3. El cálculo debe ser determinista cuando sea posible**

Los modelos de lenguaje son útiles para interpretar lenguaje, pero las operaciones numéricas deben delegarse a componentes especializados.

**4. La trazabilidad forma parte del diseño**

Un sistema no solo debe responder; debe permitir investigar qué ocurrió cuando la respuesta o el proceso no fueron los esperados.

**5. Los límites también son parte del producto**

Definir explícitamente lo que el sistema puede hacer y lo que no puede hacer evita expectativas incorrectas y facilita su evolución.

**6. La automatización encuentra lo que la validación manual no cubre**

La batería manual de 31/31 dio por bueno un sistema en el que todo filtro compuesto con `&` estaba roto — nadie había probado esa combinación. La suite automatizada lo detectó en su primer run, junto con una ambigüedad de routing que la muestra manual no había revelado. Las pruebas manuales validan lo que se te ocurrió probar; la automatización prueba el espacio.

---

## 📄 Licencia

MIT License — ver [LICENSE](LICENSE).

## 🙏 Créditos

**Stack:** Streamlit · Pandas · NumPy · LangChain · Groq · LangSmith · Matplotlib · Seaborn
**Dataset de ejemplo:** datos de demostración

© 2026 — Asistente de Análisis de Datos con IA

---

### Estado del proyecto

```
Batería funcional manual:    31/31
Evaluation Suite v2:         17/17 (A 6/6 · B 6/6 · C 2/2 · D 3/3)
pytest sin LLM (CI-ready):   44/44
        ↓
Versión validada: cálculo determinista, seguridad AST testeada,
routing verificado con telemetría (first-valid 100%, retry proxy)
        ↓
Evolución prevista:
  · retry_rate real (API de traces de LangSmith)
  · tool-calling nativo (elimina la fricción de formato ReAct)
  · JSON estructurado en la presentación → asserts exactos
  · refactor: tools sin st.* → pipeline testeable sin Streamlit
  · CI con GitHub Actions (badge en el README)
```

**Última actualización:** 2026-09-05 — tras el fix del routing (A2), la validación de la suite completa (17/17) y el hardening de seguridad.
