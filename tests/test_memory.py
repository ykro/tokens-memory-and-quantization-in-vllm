"""The memory math on its own, then every experiment against the built-in mock servers."""

import importlib.util
from pathlib import Path

import pytest

from vllm_lab import hf_facts, kv_cache_info, memory, mock_server, tokenize
from vllm_lab.quiz import QUIZ, is_correct

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def servers():
    started = {name: mock_server.start(profile=name) for name in ("gemma", "gemma-qat", "qwen")}
    yield {name: url for name, (_, url) in started.items()}
    for srv, _ in started.values():
        srv.shutdown()


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_measured_budget_adds_up():
    m = memory.MEASURED_E4B
    assert m["kv_cache_gib"] == pytest.approx(3.43)
    assert round(memory.L4_USABLE_GIB * memory.GPU_MEMORY_UTILIZATION, 2) == m["budget_gib"]
    params = hf_facts.find_model("google/gemma-4-E4B-it")["params_total"]
    room = memory.kv_room_gib(memory.weights_gib(params, "bf16"))
    assert room == pytest.approx(m["kv_cache_gib"], abs=0.02)


def test_textbook_kv_formula():
    cfg = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128,
           "num_attention_heads": 32, "hidden_size": 4096}
    shape = memory.kv_shape(cfg)
    assert shape.naive_bytes_per_token == 2 * 32 * 8 * 128 * 2 == shape.full_bytes_per_token


def test_gemma4_nested_config_with_windows_and_shared_layers():
    shape = memory.kv_shape(hf_facts.load_config("google/gemma-4-E4B-it", download=False))
    assert shape.shared_layers == 18 and shape.full_layers + shape.sliding_layers == 24
    assert shape.sliding_window == 512
    assert shape.full_bytes_per_token < shape.naive_bytes_per_token
    # past the window, only the full-attention layers keep growing
    assert shape.request_bytes(4096) - shape.request_bytes(2048) == \
        shape.full_bytes_per_token * 2048


def test_qwen_linear_layers_add_no_per_token_cache():
    shape = memory.kv_shape(hf_facts.load_config("Qwen/Qwen3.5-4B", download=False))
    assert shape.linear_layers == 24 and shape.full_layers == 8


def test_quiz_grading_ignores_case_and_accents():
    assert is_correct("Marte.", ("marte",)) and is_correct("La respuesta es 42", ("42",))
    assert not is_correct("142", ("42",)) and len(QUIZ) == 10


def test_mock_tokenize_and_metrics(servers):
    assert tokenize(servers["gemma"], "gemma", "hola mundo") > 0
    bf16, qat = kv_cache_info(servers["gemma"]), kv_cache_info(servers["gemma-qat"])
    assert qat["capacity_tokens"] > bf16["capacity_tokens"] > 0


def test_01_weights_grow_and_int4_is_smaller(servers):
    rows = load("01_weights_math").run(servers["gemma"])["rows"]
    assert [r["bf16_gib"] for r in rows] == sorted(r["bf16_gib"] for r in rows)
    assert all(r["int4_gib"] < r["bf16_gib"] for r in rows)
    assert not next(r for r in rows if r["id"].endswith("31B-it"))["fits_l4"]["bf16"]


def test_02_kv_per_token(servers):
    r = load("02_kv_cache_per_token").run(servers["gemma"])
    assert all(row["real_kib_per_token"] > 0 for row in r["rows"])
    assert r["vllm_kv_cache"]["capacity_tokens"] > 0


def test_03_longer_requests_fewer_at_once(servers):
    conc = [row["concurrent_real"] for row in
            load("03_max_model_len_tradeoff").run(servers["gemma"])["rows"]]
    assert conc == sorted(conc, reverse=True) and conc[0] > conc[-1]


def test_04_tokenizers_differ(servers):
    r = load("04_tokenizer_comparison").run(servers["gemma"], servers["qwen"])
    assert r["gemma"]["es"]["repeatable"]
    assert r["gemma"]["code"]["tokens"] != r["qwen"]["code"]["tokens"]


def test_05_quantized_saves_memory_within_tolerance(servers):
    r = load("05_quantization_quality").run(servers["gemma"], servers["gemma-qat"])
    assert r["weights_saved_gib"] > 0 and r["kv_capacity_gain_x"] > 1
    assert r["accuracy_drop"] <= r["tolerance"]


def test_ladder_reads_the_saved_file_and_keeps_real_numbers():
    import verify_memory

    r5 = {"bf16": {"weights": "/models/google/gemma-4-E4B-it", "accuracy": 1.0,
                   "weights_file_gib": 1.0, "kv_capacity_tokens": 1},
          "quantized": {"weights": "/models/google/gemma-4-E4B-it-qat-w4a16-ct", "accuracy": 1.0,
                        "weights_file_gib": 1.0, "kv_capacity_tokens": 2},
          "extras": []}
    ladder = verify_memory.build_ladder(r5)
    assert [m["model"] for m in ladder] == ["E2B", "E4B", "E4B int4", "12B int4"]
    # the saved file wins over this run's bf16 server; the run only fills what the file lacks
    assert ladder[1]["kv_capacity_tokens"] == 93138 and ladder[2]["kv_capacity_tokens"] == 2
    assert ladder[0]["kv_capacity_tokens"] > ladder[1]["kv_capacity_tokens"] > ladder[3]["kv_capacity_tokens"]
