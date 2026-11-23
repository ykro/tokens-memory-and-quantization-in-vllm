"""Each snippet in practica/ and its 'prueba esto' variant runs without a GPU and matches the repo facts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vllm_lab import mock_server

ROOT = Path(__file__).resolve().parent.parent
PRACTICA = ROOT / "practica"


@pytest.fixture(scope="module")
def port():
    server, _ = mock_server.start(profile="gemma")
    yield server.server_address[1]
    server.shutdown()


def run_script(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(path)], cwd=ROOT, capture_output=True, text=True, timeout=60, check=False
    )


def run_variant(name: str, old: str, new: str, tmp_path: Path, port: int | None = None) -> subprocess.CompletedProcess:
    src = (PRACTICA / name).read_text()
    assert old in src
    src = src.replace(old, new, 1)
    if port is not None:
        src = src.replace("localhost:8000", f"localhost:{port}")
    copy = tmp_path / name
    copy.write_text(src)
    return run_script(copy)


def test_01_pesos_bf16_and_variants(tmp_path):
    r = run_script(PRACTICA / "01_pesos.py")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "gemma-4-E4B-it  8.00B en bf16: 14.89 GiB (cabe)",
        "gemma-4-12B-it 11.96B en bf16: 22.28 GiB (no cabe)",
        "checkpoint int4 real de E4B: 10.72 GiB (L4 utilizable: 22.03 GiB)",
    ]
    fp8 = run_variant("01_pesos.py", 'dtype = "bf16"', 'dtype = "fp8"', tmp_path)
    assert fp8.returncode == 0 and "11.14 GiB (cabe)" in fp8.stdout
    m31 = run_variant("01_pesos.py", '"google/gemma-4-12B-it"', '"google/gemma-4-31B-it"', tmp_path)
    assert m31.returncode == 0 and "31.27B en bf16: 58.25 GiB (no cabe)" in m31.stdout


def test_02_kv_roto_matches_saved_vllm_error():
    r = run_script(PRACTICA / "02_kv_roto.py")
    assert r.returncode != 0
    saved = (ROOT / "benchmarks" / "does_not_fit.txt").read_text()
    expected = "2.18 GiB KV cache is needed, which is larger than the available KV cache memory (1.23 GiB)"
    assert expected in saved and expected in r.stderr
    assert "estimated maximum model length is 69008" in r.stderr


def test_02_kv_presupuesto_and_variants(tmp_path):
    r = run_script(PRACTICA / "02_kv_presupuesto.py")
    assert r.returncode == 0, r.stderr
    assert "presupuesto 20.27 GiB -> KV-cache: 3.43 GiB (16 KiB/tok)" in r.stdout
    assert "arranca: pide 2.18 GiB, caben 1.58 peticiones" in r.stdout

    short = run_variant(
        "02_kv_presupuesto.py",
        "utilizacion, max_model_len = 0.92, 131072",
        "utilizacion, max_model_len = 0.92, 8192",
        tmp_path,
    )
    assert short.returncode == 0 and "pide 0.30 GiB, caben 11.36 peticiones" in short.stdout

    cap69k = run_variant(
        "02_kv_roto.py",
        "utilizacion, max_model_len = 0.82, 131072",
        "utilizacion, max_model_len = 0.82, 69008",
        tmp_path,
    )
    assert cap69k.returncode == 0 and "pide 1.23 GiB, caben 1.00 peticiones" in cap69k.stdout


def test_03_concurrencia_and_variants(tmp_path):
    r = run_script(PRACTICA / "03_concurrencia.py")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "KV-cache real en L4 -> bf16: 93,138 tokens | int4: 198,111 tokens",
        " 2,048 tokens/petición -> bf16: 45 a la vez | int4: 96 a la vez",
        " 8,192 tokens/petición -> bf16: 11 a la vez | int4: 24 a la vez",
        "32,768 tokens/petición -> bf16:  2 a la vez | int4:  6 a la vez",
    ]
    long_req = run_variant(
        "03_concurrencia.py",
        "largos = (2048, 8192, 32768)",
        "largos = (2048, 8192, 65536)",
        tmp_path,
    )
    assert long_req.returncode == 0
    assert "65,536 tokens/petición -> bf16:  1 a la vez | int4:  3 a la vez" in long_req.stdout


def test_04_tokenizar_and_variants(port, tmp_path):
    r = run_variant("04_tokenizar.py", "localhost:8000", f"localhost:{port}", tmp_path)
    assert r.returncode == 0, r.stderr
    assert "Gemma real -> en 1.12 (7,281), es 1.31 (6,251), code 2.80" in r.stdout
    assert "en   ->" in r.stdout and "es   ->" in r.stdout and "code ->" in r.stdout

    custom_json = run_variant(
        "04_tokenizar.py",
        "\"code\": \"def ticket(d): return {'cambio': True, 'reembolso': False}\"",
        "'json': '{\"ticket\": 409, \"producto\": \"audífonos\", \"cambio\": true}'",
        tmp_path,
        port=port,
    )
    assert custom_json.returncode == 0 and "json ->" in custom_json.stdout
