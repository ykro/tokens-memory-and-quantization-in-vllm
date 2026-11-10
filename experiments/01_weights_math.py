"""01. Weights first: parameters x bytes per number, against the 24 GB of an L4.

Every parameter is a number, and every number takes bytes: 2 in bf16, 1 in fp8, half a
byte in int4. Multiply by the parameter count (read from the Hugging Face API, checked in
at benchmarks/models.json) and you know whether a model even starts on one GPU.

Two surprises show up. Gemma 4 "E4B" stores about 8 billion parameters: the "E" means
effective, 4B do the work on each token, but the file also carries per-layer embeddings
and the image and audio encoders, and all of it has to sit on the GPU. And the published
int4 checkpoints are bigger than the int4 math predicts, because only the big matrices of
the text layers are quantized; embeddings and encoders stay in bf16.

    python experiments/01_weights_math.py          # live server (tunnel open)
    python experiments/01_weights_math.py --mock   # no GPU
"""

from __future__ import annotations

from vllm_lab import cli, hf_facts, memory, served_model


def run(base_url: str) -> dict:
    served = served_model(base_url).get("root")
    facts = hf_facts.load_models()
    usable = memory.L4_USABLE_GIB
    rows = []
    for m in facts["models"]:
        if m["quantized"]:
            continue
        row = {"id": m["id"], "params_b": round(m["params_total"] / 1e9, 2)}
        for dtype in memory.BYTES_PER_DTYPE:
            w = memory.weights_gib(m["params_total"], dtype)
            row[f"{dtype}_gib"] = round(w, 2)
            row[f"{dtype}_kv_room_gib"] = round(memory.kv_room_gib(w), 2)
        row["fits_l4"] = {d: row[f"{d}_kv_room_gib"] > 0 for d in memory.BYTES_PER_DTYPE}
        row["served_now"] = bool(served) and served.endswith(m["id"])
        rows.append(row)
    rows.sort(key=lambda r: r["params_b"])
    quantized = []
    for m in facts["models"]:
        if not m["quantized"]:
            continue
        base = hf_facts.find_model(m["id"].split("-qat")[0])
        quantized.append({
            "id": m["id"],
            "file_gib": round(m["safetensors_bytes"] / memory.GIB, 2),
            "bf16_file_gib": round(base["safetensors_bytes"] / memory.GIB, 2),
            "pure_int4_math_gib": round(memory.weights_gib(base["params_total"], "int4"), 2),
            "saved_vs_bf16": round(1 - m["safetensors_bytes"] / base["safetensors_bytes"], 2),
        })
    return {
        "checked": facts["checked"],
        "gpu": {"card_gb": memory.L4_NOMINAL_GB, "usable_gib": usable,
                "budget_gib": memory.L4_BUDGET_GIB,
                "gpu_memory_utilization": memory.GPU_MEMORY_UTILIZATION},
        "reserved_besides_weights_gib": round(memory.NON_TORCH_GIB +
                                              memory.ACTIVATION_RESERVE_GIB, 2),
        "served_weights": served,
        "rows": rows,
        "quantized_checkpoints": quantized,
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    cli.show("One NVIDIA L4", result["gpu"])
    print(f"\n{'model':26} {'params':>8} {'bf16':>9} {'fp8':>9} {'int4':>9}   fits in bf16?")
    for r in result["rows"]:
        mark = "  <- on the server" if r["served_now"] else ""
        print(f"{r['id'].split('/')[1]:26} {r['params_b']:>7.2f}B {r['bf16_gib']:>6.2f} GiB "
              f"{r['fp8_gib']:>5.2f} GiB {r['int4_gib']:>5.2f} GiB   "
              f"{'yes' if r['fits_l4']['bf16'] else 'no'}{mark}")
    cli.show("Published int4 checkpoints vs the int4 math", {"rows": result["quantized_checkpoints"]})


if __name__ == "__main__":
    main()
