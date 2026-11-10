"""Facts about each model, read once from the public Hugging Face API and checked in.

Two sources, both public and free, no token needed:

* https://huggingface.co/api/models/<id>            -> `safetensors.total` (parameter count)
* https://huggingface.co/api/models/<id>/tree/main  -> size in bytes of each .safetensors file
* https://huggingface.co/<id>/resolve/main/config.json -> layers, heads, head size

The experiments read the checked-in copies (`benchmarks/models.json`, `benchmarks/configs/`)
so they run offline and give the same numbers every time. To refresh them:

    python -m vllm_lab.hf_facts
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
MODELS_JSON = BENCH / "models.json"
CONFIGS = BENCH / "configs"

# Full-precision (bf16) checkpoints compared in experiments 01 and 02.
MODEL_IDS = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.5-4B",  # newest general Qwen instruct model of 4-8B on the Hub (2026-11-10)
]
# Quantized checkpoints published by Google (int4 weights, bf16 activations).
QUANTIZED_IDS = [
    "google/gemma-4-E4B-it-qat-w4a16-ct",
    "google/gemma-4-12B-it-qat-w4a16-ct",
]


def _get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-lab/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def config_path(model_id: str) -> Path:
    return CONFIGS / (model_id.replace("/", "__") + ".json")


def load_config(model_id: str, download: bool = True) -> dict:
    """The model's config.json, from the local cache or downloaded once into it."""
    path = config_path(model_id)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        CONFIGS.mkdir(parents=True, exist_ok=True)
        cfg = _get_json(f"https://huggingface.co/{model_id}/resolve/main/config.json")
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    return json.loads(path.read_text())


def load_models() -> dict:
    return json.loads(MODELS_JSON.read_text())


def find_model(model_id: str | None) -> dict | None:
    """Looks up a model by id; also accepts a local path like /models/google/gemma-4-E4B-it."""
    if not model_id:
        return None
    for row in load_models()["models"]:
        if model_id == row["id"] or model_id.endswith("/" + row["id"]):
            return row
    return None


def fetch_row(model_id: str, quantized: bool) -> dict:
    info = _get_json(f"https://huggingface.co/api/models/{model_id}")
    tree = _get_json(f"https://huggingface.co/api/models/{model_id}/tree/main")
    files = sum(f.get("size", 0) for f in tree if f["path"].endswith(".safetensors"))
    st = info.get("safetensors") or {}
    return {
        "id": model_id,
        "family": model_id.split("/")[0].lower(),
        "quantized": quantized,
        "params_total": st.get("total"),
        "params_by_dtype": st.get("parameters"),
        "safetensors_bytes": files,
        "license": (info.get("cardData") or {}).get("license"),
        "created_at": info.get("createdAt"),
    }


def refresh() -> dict:
    rows = [fetch_row(m, False) for m in MODEL_IDS] + [fetch_row(m, True) for m in QUANTIZED_IDS]
    payload = {
        "checked": date.today().isoformat(),
        "source": "https://huggingface.co/api/models/<id> (safetensors.total) y "
                  "https://huggingface.co/api/models/<id>/tree/main (tamaño de los .safetensors)",
        "note": "params_total cuenta todo lo que trae el archivo: capas de texto, embeddings y, "
                "en Gemma 4 y Qwen 3.5, los encoders de imagen y audio. En los checkpoints "
                "int4 (I32) cada número de 32 bits empaca varios pesos de 4 bits, así que "
                "para memoria hay que leer safetensors_bytes.",
        "models": rows,
    }
    BENCH.mkdir(exist_ok=True)
    MODELS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for model_id in MODEL_IDS + QUANTIZED_IDS[:1]:
        config_path(model_id).unlink(missing_ok=True)
        load_config(model_id)
    return payload


if __name__ == "__main__":
    data = refresh()
    for row in data["models"]:
        print(f"{row['id']:40} {row['params_total']:>15,} params "
              f"{row['safetensors_bytes'] / 1e9:6.2f} GB on disk")
    print(f"Saved {MODELS_JSON.relative_to(ROOT)} and {CONFIGS.relative_to(ROOT)}/")
