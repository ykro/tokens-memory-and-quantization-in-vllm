# Tokens, memoria y cuantización en vLLM

**¿Por qué un modelo de 4B no cabe donde uno cree?**

- **Por qué es relevante.** La primera vez que levantas un modelo propio, la cuenta mental suele ser «4 mil millones de parámetros, 2 bytes cada uno, 8 GB, sobra espacio en una GPU de 24 GB». Después vLLM arranca, reserva casi toda la tarjeta y te dice que solo caben unas cuantas peticiones largas a la vez. Este repo hace la cuenta completa con números reales: pesos, activaciones, KV-cache, tokens y cuantización.
- **Para quién es.** Para quien sabe Python y ya llamó alguna API de IA (OpenAI, Gemini, Claude), pero nunca sirvió un modelo propio. No necesitas saber de machine learning, JAX ni PyTorch.
- **Quién debería verlo.** Quien tiene que decidir qué GPU pedir, qué modelo cabe en ella, cuánto contexto prometer a sus usuarios o si vale la pena una versión cuantizada.
- **Ruta completa.** Forma parte del [vLLM for Software Devs Field Guide](https://github.com/ykro/vllm-for-software-devs-field-guide), servir modelos abiertos con vLLM. Viene después de [Tu primer servidor vLLM con Gemma](https://github.com/ykro/first-vllm-server-with-gemma) y usa el mismo servidor en Cloud Run.

---

## Modelo mental

### 1. La GPU se reparte en tres partes, y los pesos son solo la primera

vLLM toma una fracción de la memoria de la GPU (`--gpu-memory-utilization`, 0.92 en esta serie), carga los pesos, mide cuánto necesita para las activaciones de un forward pass y le da todo lo demás al KV-cache. Medido en el servidor real de esta unidad (vLLM 0.33.0, `google/gemma-4-E4B-it` en bf16, Cloud Run con una NVIDIA L4):

| Concepto | GiB |
| :--- | ---: |
| Memoria utilizable que reporta la GPU (la tarjeta dice 24 GB, que son 22.35 GiB) | 22.03 |
| Presupuesto de vLLM con `gpu-memory-utilization 0.92` | 20.27 |
| Pesos más memoria fuera de PyTorch (contexto de CUDA y similares) | 15.38 |
| Pico de activaciones | 1.46 |
| **Lo que queda para el KV-cache** | **3.43** |

El modelo se llama E4B, pero su archivo trae **8.0 mil millones de parámetros** (`safetensors.total` en la API de Hugging Face). La «E» significa *effective*: unos 4B trabajan en cada token, y el resto son embeddings por capa y los encoders de imagen y audio, que también tienen que estar en la GPU. Esa es la primera razón por la que un «4B» no cabe donde uno cree.

### 2. Cada token cuesta memoria mientras la petición está viva

Para no releer todo el prompt en cada paso del decode, el modelo guarda una *key* y un *value* por token en cada capa de atención. La fórmula de libro es

```text
bytes por token = 2 x capas x kv_heads x head_dim x 2 bytes (bf16)
```

Para Gemma 4 E4B da 84 KiB por token. Gemma 4 promete pagar bastante menos gracias a tres trucos de arquitectura que el código lee de cada `config.json`. Las capas de ventana deslizante solo recuerdan los últimos 512 tokens, así que su cache deja de crecer. Las últimas 18 capas reutilizan el cache de capas anteriores. Solo 4 capas de atención global siguen creciendo con cada token, a 16 KiB por token. Qwen 3.5 usa otro truco: 3 de cada 4 capas usan atención lineal, con un resumen de tamaño fijo por petición.

La cuenta de libro, sin embargo, resultó optimista. Con esos 3.43 GiB mi estimación decía unos 223,500 tokens, y vLLM reporta **93,138 tokens** (y 11.37 peticiones simultáneas de 8,192 tokens). El número del servidor es el que vale: lo publica en `/metrics` como `kv_cache_size_tokens` y en su log de arranque. Además cambia con `--max-model-len`, porque vLLM reparte el cache en grupos de capas, y con `--max-model-len 131072` el mismo cache de 3.43 GiB se reporta como 206,728 tokens (1.58 peticiones de 131,072 tokens).

### 3. Contexto largo o muchos usuarios, el trade-off de `--max-model-len`

El KV-cache es una sola bolsa. Cada petición activa toma de ella en proporción a su largo (prompt más respuesta). Con la misma bolsa caben muchas peticiones de 2k tokens o pocas de 32k. `--max-model-len` es la promesa de aceptar peticiones hasta ese largo; vLLM no arranca si la bolsa no alcanza ni para una, y cuando se llena, las peticiones nuevas esperan en cola. Medido: con `--max-model-len 131072` y `--gpu-memory-utilization 0.82`, vLLM se detuvo con `2.18 GiB KV cache is needed, which is larger than the available KV cache memory (1.23 GiB)` y sugirió un máximo de 69,008 tokens (`benchmarks/does_not_fit.txt`).

### 4. La cuantización libera memoria, y hay que medir cuánta calidad cuesta

Guardar cada peso en 4 bits en lugar de 16 reduce la matemática pura a la cuarta parte. El checkpoint int4 que publica Google (`google/gemma-4-E4B-it-qat-w4a16-ct`) pesa 10.72 GiB contra 14.89 GiB en bf16, porque solo se cuantizan las matrices grandes de las capas de texto; los embeddings y los encoders se quedan en bf16. Esos 4 GiB liberados se van directo al KV-cache. «qat» significa que el modelo se entrenó sabiendo que iba a ser cuantizado, y «w4a16» que los pesos van en 4 bits y las activaciones en 16.

---

## Los cinco experimentos

| # | Archivo | Pregunta | Qué mide |
| :--- | :--- | :--- | :--- |
| 01 | `experiments/01_weights_math.py` | ¿Cuánto pesan los pesos? | Parámetros por bytes en bf16, fp8 e int4 para Gemma 4 E2B, E4B, 12B, 31B y Qwen3.5-4B, contra los 24 GB de una L4 |
| 02 | `experiments/02_kv_cache_per_token.py` | ¿Cuánto cuesta cada token? | KV-cache por token con la fórmula de libro y con la arquitectura real; cuántos tokens caben; la capacidad real que reporta vLLM en `/metrics` |
| 03 | `experiments/03_max_model_len_tradeoff.py` | ¿Contexto largo o muchos usuarios? | Con la bolsa fija de 3.43 GiB, cuántas peticiones de 2k, 8k y 32k tokens caben a la vez y cuánto cuesta cada lugar por hora |
| 04 | `experiments/04_tokenizer_comparison.py` | ¿Cuántos tokens es mi texto? | El mismo texto en español, inglés y código por `POST /tokenize`; tokens por palabra en Gemma y en Qwen |
| 05 | `experiments/05_quantization_quality.py` | ¿Cuánta calidad cuesta int4? | 10 preguntas cerradas a un servidor bf16 y a uno int4; exactitud, memoria de pesos y capacidad de KV-cache |

Los tamaños de los modelos salen de la API pública de Hugging Face (`https://huggingface.co/api/models/<id>`, campo `safetensors.total`, y el tamaño de los archivos) y están guardados en `benchmarks/models.json` con la fecha de consulta. Los `config.json` están en `benchmarks/configs/`. Para refrescarlos, `make facts`.

---

## Cómo correrlo

### Sin GPU, en tu laptop

Cada experimento tiene `--mock`, que levanta servidores falsos deterministas con las mismas rutas HTTP que vLLM (`/v1/models`, `/v1/chat/completions`, `/tokenize` y `/metrics`). Sirve para entender el código y para la integración continua; sus números siguen la aritmética de la unidad, pero no son mediciones.

```bash
git clone https://github.com/ykro/tokens-memory-and-quantization-in-vllm.git
cd tokens-memory-and-quantization-in-vllm
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make all                                         # ruff + pytest + juez con --mock
python experiments/01_weights_math.py --mock     # cualquier experimento por separado
```

### Con el servidor real en Cloud Run

vLLM necesita una GPU NVIDIA, así que el servidor corre en Cloud Run con una L4 on-demand (`min-instances 0`). Necesitas un proyecto de Google Cloud con cuota de GPU L4 en Cloud Run.

```bash
export GCP_PROJECT_ID=<tu-proyecto>

make stage && make stage-quant      # una vez: copia los pesos a un bucket en la misma región
make deploy                         # vllm-gemma, Gemma 4 E4B en bf16
make deploy-quant                   # vllm-gemma-qat, el checkpoint int4
```

Cada servicio despierto ocupa una L4 de tu cuota regional. En otra terminal, abre un túnel por servicio (sin URL pública ni API key en el código):

```bash
gcloud run services proxy vllm-gemma --region us-east4 --port 8000       # make tunnel
gcloud run services proxy vllm-gemma-qat --region us-east4 --port 8001   # make tunnel-quant
```

Y corre el juez contra los servidores reales:

```bash
make verify-live      # escribe benchmarks/latest_results.json y benchmarks/terminal_output.txt
make report           # lo mismo, imprimiendo además el resultado crudo de cada experimento
make report-saved     # reimprime el reporte de la última corrida en vivo, sin servidores ni GPU
```

En la fila 03 del juez, `vLLM 45/11/2 at 2k/8k/32k` son las peticiones que caben según la capacidad en tokens que reportó vLLM; `estimate 67/23/6` es la cuenta con el `config.json`, que resultó 2.4 veces optimista.

Para comparar tokenizadores con Qwen, levanta un tercer servicio y pásale su URL al juez:

```bash
make deploy-qwen && make tunnel-qwen
make verify-live QWEN_BASE_URL=http://localhost:8002/v1
```

Para alinear más modelos en el experimento 05 (cada uno ocupa otra L4 mientras está despierto), despliega `make deploy-e2b` o `make deploy-12b`, abre su túnel y pásalos como `EXTRA_ARGS="--extra-base-url http://localhost:8003/v1"`; cada corrida los suma a `benchmarks/memory_ladder.json`. Para ver el error de un contexto que no cabe, `make deploy-toolong` (131,072 tokens con `--gpu-memory-utilization 0.82`): el despliegue falla a propósito y el motivo queda en el log del servicio.

Al terminar, `make teardown` borra todos los servicios de la unidad. Con `min-instances 0` un servicio sin tráfico no cobra GPU, pero borrarlo es la única forma de estar seguro.

---

## Resultados

Corrida en vivo del 2026-11-10 con `make verify-live QWEN_BASE_URL=http://localhost:8002/v1` (vLLM 0.33.0, una NVIDIA L4 de Cloud Run en `us-east4` por servicio, pesos desde un bucket). Todo queda en `benchmarks/latest_results.json`, `benchmarks/terminal_output.txt`, `benchmarks/memory_ladder.json` y `benchmarks/does_not_fit.txt`. El juez pasa los 8 checks: los 7 de los cinco experimentos y la escalera de modelos (E2B, E4B, E4B int4 y 12B int4 en el mismo quiz), que lee `benchmarks/memory_ladder.json`.

### Memoria de la GPU, del log de arranque de vLLM

La L4 reporta 22.03 GiB utilizables; con `--gpu-memory-utilization 0.92` el presupuesto es 20.27 GiB.

| Servidor | Pesos y CUDA | Activaciones | KV-cache | Tokens en el KV-cache | Peticiones de 8,192 a la vez |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E2B, bf16 | 10.12 GiB | 1.40 GiB | 8.76 GiB | 717,880 | 87.6 |
| **Gemma 4 E4B, bf16** | 15.38 GiB | 1.46 GiB | **3.43 GiB** | 93,138 | 11.4 |
| Gemma 4 E4B, int4 (`qat-w4a16-ct`) | 11.45 GiB | 1.52 GiB | **7.30 GiB** | 198,111 | 24.2 |
| Gemma 4 12B, int4 (`qat-w4a16-ct`) | 8.69 GiB | 1.54 GiB | 10.04 GiB | 41,851 | 5.1 |
| Qwen 3.5 4B, bf16 | 8.87 GiB | 2.06 GiB | 9.34 GiB | 215,598 | 26.3 |

Gemma 4 12B en bf16 pesa 22.28 GiB, más que los 22.03 GiB utilizables: no cabe (cuenta, no se desplegó).

### Los cinco experimentos

| Experimento | Resultado en L4 |
| :--- | :--- |
| **01. Pesos** | E4B bf16 14.89 GiB, fp8 7.45, int4 3.72. 12B bf16 22.28 GiB y 31B 58.25 GiB no caben. El checkpoint int4 de Google de E4B pesa 10.72 GiB (no 3.72, porque embeddings y encoders siguen en bf16) |
| **02. KV-cache por token** | Fórmula de libro 84 KiB por token; con la arquitectura 16 KiB por token más una ventana de 20 MiB por petición. vLLM: 93,138 tokens, contra 223,508 de la estimación (2.4 veces menos) |
| **03. `max-model-len`** | Con 93,138 tokens de capacidad caben 45, 11 y 2 peticiones de 2,048, 8,192 y 32,768 tokens (la estimación daba 67, 23 y 6) |
| **04. Tokenizador** | Gemma: español 1.31, inglés 1.12, código 2.80 tokens por palabra (español 1.17 veces el inglés). Qwen: 1.38, 1.12 y 2.60. En 8,192 tokens caben 6,251 palabras en español y 7,281 en inglés con Gemma |
| **05. Cuantización** | Los pesos bajan 4.17 GiB (14.89 a 10.72), el KV-cache sube 2.13 veces (93,138 a 198,111 tokens) y las 10 preguntas siguen en 10 de 10 con bf16, int4, E2B y 12B int4 |

### Cómo leer estas cifras

- **GB contra GiB.** Las tarjetas se venden en GB (10⁹ bytes) y vLLM reporta en GiB (2³⁰ bytes). 24 GB son 22.35 GiB.
- **Estimación contra medición.** Los experimentos 02 y 03 calculan a partir de cada `config.json` y del presupuesto medido de E4B. Para los demás modelos suponen el mismo pico de activaciones (1.46 GiB) y la misma memoria fuera de PyTorch (0.49 GiB). La capacidad que reporta vLLM en `/metrics` es la referencia.
- **Capacidad contra estimación.** La capacidad en tokens del servidor depende de `--max-model-len` (93,138 tokens a 8,192 y 206,728 a 131,072 con el mismo cache). Las filas de 2k y 32k del experimento 03 dividen la capacidad del servidor entre el largo de la petición, así que son una aproximación; la de 8,192 es la que publica vLLM (`kv_cache_max_concurrency`).
- **Concurrencia.** El experimento 03 cuenta solo el KV-cache. vLLM además limita las peticiones activas con `--max-num-seqs`, y el servidor de la serie arranca con `--max-model-len 8192`, así que la fila de 32k es una pregunta de diseño (qué pasaría si subieras ese límite).
- **Diez preguntas.** Detectan caídas grandes de calidad. Para tu aplicación, evalúa con tus propios datos.

---

## Práctica

La carpeta `practica/` contiene los fragmentos cortos del video de práctica, cubiertos junto con sus variantes de «prueba esto» en `tests/test_practica.py`. Todos corren en CPU sin GPU: leen los datos guardados en `benchmarks/` (`models.json`, `configs/`, `latest_results.json` y `does_not_fit.txt`) y hablan con el servidor falso de `vllm_lab/mock_server.py`. Se corren desde la raíz del repositorio con Python 3.11 o más nuevo.

```bash
python3 -m vllm_lab.mock_server &       # servidores falsos en localhost:8000..8002, en segundo plano
curl -s localhost:8000/metrics | grep cache_config_info

python3 practica/01_pesos.py            # parámetros reales en GiB (2^30) contra los 22.03 GiB de la L4
python3 practica/02_kv_roto.py          # falla a propósito: 131,072 tokens con utilización 0.82 pide 2.18 GiB y solo quedan 1.23 GiB
cat benchmarks/does_not_fit.txt         # el mismo ValueError medido en la L4 real
python3 practica/02_kv_presupuesto.py   # con utilización 0.92 quedan 3.43 GiB de KV-cache y sí arranca
python3 practica/03_concurrencia.py     # conversaciones simultáneas de 2k, 8k y 32k tokens en bf16 (93,138) e int4 (198,111)
python3 practica/04_tokenizar.py        # POST /tokenize en inglés, español y código contra las tasas reales de Gemma

python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[dev]"      # openai, pytest y ruff
make all                                # ruff, 17 tests y el juez contra los servidores falsos
cat benchmarks/terminal_output.txt      # el juez medido en la L4 real (2026-11-10)
kill %1                                 # apaga los servidores falsos
```

---

## Estructura del repositorio

```text
.
├── benchmarks/
│   ├── configs/                      # config.json de cada modelo, descargado de Hugging Face
│   ├── models.json                   # parámetros y tamaño en disco, con fecha de consulta
│   ├── latest_results.json           # lo escribe make verify-live
│   ├── terminal_output.txt           # salida del juez en vivo
│   ├── memory_ladder.json            # E2B, E4B y 12B int4 en el mismo quiz; el juez lo lee
│   ├── does_not_fit.txt              # el error de vLLM con un contexto que no cabe
│   └── pricing.json                  # precio por hora de la instancia de Cloud Run con L4
├── diagrams/                         # 3 diagramas blue-gold (html, svg, png)
├── experiments/
│   ├── 01_weights_math.py            # parámetros x bytes por dtype contra una L4
│   ├── 02_kv_cache_per_token.py      # KV-cache por token y cuántos tokens caben
│   ├── 03_max_model_len_tradeoff.py  # contexto largo contra peticiones simultáneas
│   ├── 04_tokenizer_comparison.py    # español, inglés y código por POST /tokenize
│   └── 05_quantization_quality.py    # bf16 contra int4, memoria contra exactitud
├── practica/                         # fragmentos cortos del video de práctica (sin GPU)
│   ├── 01_pesos.py                   # GiB reales de E4B, 12B y el checkpoint int4
│   ├── 02_kv_roto.py                 # max_model_len 131072 con utilización 0.82 (ValueError)
│   ├── 02_kv_presupuesto.py          # arreglo en una línea con utilización 0.92
│   ├── 03_concurrencia.py            # conversaciones simultáneas en bf16 vs int4
│   └── 04_tokenizar.py               # POST /tokenize en inglés, español y código
├── gcp/
│   ├── deploy_vllm.sh                # un servicio de Cloud Run con L4 por modelo
│   ├── stage_weights.sh              # pesos de Hugging Face a Cloud Storage, una vez
│   └── teardown.sh                   # borra los servicios
├── tests/
│   ├── test_memory.py                # la aritmética y los cinco experimentos contra el mock
│   └── test_practica.py              # los fragmentos de practica/ y sus variantes «prueba esto»
├── vllm_lab/
│   ├── cli.py                        # --mock, URLs de los servidores y salida
│   ├── client.py                     # SDK de OpenAI, /tokenize y /metrics
│   ├── hf_facts.py                   # lee y refresca los datos de Hugging Face
│   ├── memory.py                     # la aritmética de pesos y KV-cache
│   ├── mock_server.py                # servidores falsos (bf16, int4 y Qwen)
│   └── quiz.py                       # las 10 preguntas y su calificación
├── Makefile
├── pyproject.toml
└── verify_memory.py                  # el Outside Judge
```
