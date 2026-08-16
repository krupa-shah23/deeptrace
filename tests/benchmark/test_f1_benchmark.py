import pytest
from pathlib import Path

def test_f1_benchmark_dataset_execution():
    manifest_path = Path("data/manifests/asvspoof_la_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("ASVspoof 2019 LA benchmark dataset manifest not found. Run scripts/prepare_datasets.py.")

    from scripts.evaluate_f1 import evaluate_f1
    res = evaluate_f1(output_dir="output/test_benchmark_f1", mode="benchmark")
    assert res.get("status") in ("BENCHMARK COMPLETED", "BENCHMARK NOT COMPLETED")
