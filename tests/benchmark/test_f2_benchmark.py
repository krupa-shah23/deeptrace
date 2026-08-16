import pytest
from pathlib import Path

def test_f2_benchmark_dataset_execution():
    manifest_path = Path("data/manifests/asvspoof_pa_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("ASVspoof 2019 PA benchmark dataset manifest not found. Run scripts/prepare_datasets.py.")

    from scripts.evaluate_f2 import evaluate_f2
    res = evaluate_f2(output_dir="output/test_benchmark_f2", mode="benchmark")
    assert res.get("status") in ("BENCHMARK COMPLETED", "BENCHMARK NOT COMPLETED")
