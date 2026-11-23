"""Reparto de los 22.03 GiB de la L4 y validación de max_model_len con config.json."""
import json
from pathlib import Path

t = json.loads(Path("benchmarks/configs/google__gemma-4-E4B-it.json").read_text())["text_config"]
propias = t["layer_types"][: t["num_hidden_layers"] - t["num_kv_shared_layers"]]
globales = propias.count("full_attention")
kib_tok = globales * 2 * t["num_key_value_heads"] * t["global_head_dim"] * 2 // 1024
utilizacion, max_model_len = 0.82, 131072
presupuesto = round((22563 / 1024) * utilizacion, 2)
kv_gib = round(presupuesto - 15.38 - 1.46, 2)
necesario = (max_model_len * kib_tok + 725 * 256) / 1024**2
max_estimado = (round(kv_gib * 1024**2 / 256) - 725) * 16
print(f"presupuesto {presupuesto:.2f} GiB -> KV-cache: {kv_gib:.2f} GiB ({kib_tok} KiB/tok)")
if round(necesario, 2) > kv_gib:
    raise ValueError(
        f"To serve at least one request with the model's max seq len ({max_model_len}), "
        f"({necesario:.2f} GiB KV cache is needed, which is larger than the available "
        f"KV cache memory ({kv_gib:.2f} GiB). Based on the available memory, "
        f"the estimated maximum model length is {max_estimado}."
    )
print(f"arranca: pide {necesario:.2f} GiB, caben {kv_gib / necesario:.2f} peticiones")
