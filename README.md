# Tokens, memoria y cuantización en vLLM

**¿Por qué un modelo de 4B no cabe donde uno cree?**

- **Por qué es relevante.** La primera vez que levantas un modelo propio, la cuenta mental suele ser «4 mil millones de parámetros, 2 bytes cada uno, 8 GB, sobra espacio en una GPU de 24 GB». Después vLLM arranca, reserva casi toda la tarjeta y te dice que solo caben unas cuantas peticiones largas a la vez. Este repo hace la cuenta completa con números reales: pesos, activaciones, KV-cache, tokens y cuantización.
- **Para quién es.** Para quien sabe Python y ya llamó alguna API de IA (OpenAI, Gemini, Claude), pero nunca sirvió un modelo propio. No necesitas saber de machine learning, JAX ni PyTorch.
- **Quién debería verlo.** Quien tiene que decidir qué GPU pedir, qué modelo cabe en ella, cuánto contexto prometer a sus usuarios o si vale la pena una versión cuantizada.
- **Serie completa.** Unidad C2 de la Serie C, servir modelos abiertos con vLLM. Viene después de [C1, tu primer servidor vLLM con Gemma](https://github.com/ykro/first-vllm-server-with-gemma) y usa el mismo servidor en Cloud Run.

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

Para Gemma 4 E4B da 84 KiB por token. Gemma 4 paga bastante menos gracias a tres trucos de arquitectura que el código lee de cada `config.json`. Las capas de ventana deslizante solo recuerdan los últimos 512 tokens, así que su cache deja de crecer. Las últimas 18 capas reutilizan el cache de capas anteriores. Solo 4 capas de atención global siguen creciendo con cada token, a 16 KiB por token. Qwen 3.5 usa otro truco: 3 de cada 4 capas usan atención lineal, con un resumen de tamaño fijo por petición.

### 3. Contexto largo o muchos usuarios, el trade-off de `--max-model-len`

El KV-cache es una sola bolsa. Cada petición activa toma de ella en proporción a su largo (prompt más respuesta). Con la misma bolsa caben muchas peticiones de 2k tokens o pocas de 32k. `--max-model-len` es la promesa de aceptar peticiones hasta ese largo; vLLM no arranca si la bolsa no alcanza ni para una, y cuando se llena, las peticiones nuevas esperan en cola.

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
```

Para comparar tokenizadores con Qwen, levanta un tercer servicio y pásale su URL al juez:

```bash
make deploy-qwen && make tunnel-qwen
make verify-live QWEN_BASE_URL=http://localhost:8002/v1
```

Al terminar, `make teardown` borra los tres servicios. Con `min-instances 0` un servicio sin tráfico no cobra GPU, pero borrarlo es la única forma de estar seguro.

---

## Resultados

**Pendiente de la corrida en vivo.** Esta sección se llena con la salida de `make verify-live` contra los dos servidores en Cloud Run. Las cifras de la tabla del modelo mental (22.03, 20.27, 15.38, 1.46 y 3.43 GiB) ya son mediciones del servidor real; lo que falta es la capacidad de KV-cache que reporta vLLM en `/metrics` para bf16 e int4, el conteo real de tokens del tokenizador y la exactitud de las 10 preguntas.

Lo que el juez verifica, en cualquier modo:

- Los pesos crecen con el número de parámetros, e int4 pesa menos que fp8 y fp8 menos que bf16.
- El KV-cache por token es mayor que cero en todos los modelos, y vLLM reporta capacidad.
- Con más contexto por petición caben menos peticiones a la vez.
- El tokenizador da el mismo conteo para el mismo texto.
- El modelo cuantizado ocupa menos memoria de pesos y deja más KV-cache.
- La exactitud del modelo cuantizado cae como mucho 2 de 10 preguntas respecto a bf16.

### Cómo leer estas cifras

- **GB contra GiB.** Las tarjetas se venden en GB (10⁹ bytes) y vLLM reporta en GiB (2³⁰ bytes). 24 GB son 22.35 GiB.
- **Estimación contra medición.** Los experimentos 02 y 03 calculan a partir de cada `config.json` y del presupuesto medido de E4B. Para los demás modelos suponen el mismo pico de activaciones (1.46 GiB) y la misma memoria fuera de PyTorch (0.49 GiB). La capacidad que reporta vLLM en `/metrics` es la referencia.
- **Concurrencia.** El experimento 03 cuenta solo el KV-cache. vLLM además limita las peticiones activas con `--max-num-seqs`, y el servidor de la serie arranca con `--max-model-len 8192`, así que la fila de 32k es una pregunta de diseño (qué pasaría si subieras ese límite).
- **Diez preguntas.** Detectan caídas grandes de calidad. Para tu aplicación, evalúa con tus propios datos.

---

## Estructura del repositorio

```text
.
├── benchmarks/
│   ├── configs/                      # config.json de cada modelo, descargado de Hugging Face
│   ├── models.json                   # parámetros y tamaño en disco, con fecha de consulta
│   └── pricing.json                  # precio por hora de la instancia de Cloud Run con L4
├── experiments/
│   ├── 01_weights_math.py            # parámetros x bytes por dtype contra una L4
│   ├── 02_kv_cache_per_token.py      # KV-cache por token y cuántos tokens caben
│   ├── 03_max_model_len_tradeoff.py  # contexto largo contra peticiones simultáneas
│   ├── 04_tokenizer_comparison.py    # español, inglés y código por POST /tokenize
│   └── 05_quantization_quality.py    # bf16 contra int4, memoria contra exactitud
├── gcp/
│   ├── deploy_vllm.sh                # un servicio de Cloud Run con L4 por modelo
│   ├── stage_weights.sh              # pesos de Hugging Face a Cloud Storage, una vez
│   └── teardown.sh                   # borra los servicios
├── tests/
│   └── test_memory.py                # la aritmética y los cinco experimentos contra el mock
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
