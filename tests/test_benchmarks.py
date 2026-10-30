import torch
from benchmarks.common import benchmark, write_csv


def test_benchmark_runs_on_cpu_and_records_throughput(tmp_path):
    calls = 0

    def operation():
        nonlocal calls
        calls += 1

    result = benchmark(
        operation,
        name="unit",
        device="cpu",
        dtype=torch.float32,
        warmup_iterations=2,
        measurement_iterations=3,
        details={"work_items_per_iteration": 4},
    )

    assert calls == 5
    assert result.median_latency_ms >= 0
    assert result.throughput_per_second is not None

    output = tmp_path / "results.csv"
    write_csv(output, [result])

    assert "median_latency_ms" in output.read_text(encoding="utf-8")
