from __future__ import annotations

import bench

from bench import BenchmarkResult


def test_parse_args_defaults() -> None:
    args = bench.parse_args([])

    assert args.backend == "both"
    assert args.num_seqs == 256
    assert args.max_model_len == 4096


def test_main_prints_backend_results(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        bench,
        "benchmark_inferlib",
        lambda **_: BenchmarkResult("inferlib", total_tokens=100, elapsed_seconds=2.0),
    )
    monkeypatch.setattr(
        bench,
        "benchmark_vllm",
        lambda **_: None,
    )

    exit_code = bench.main(["--backend", "both", "--num-seqs", "2", "--max-input-len", "100", "--max-output-len", "100"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "workload: backend=both" in captured.out
    assert "inferlib: Total 100tok, Time 2.00s, Throughput 50.00tok/s" in captured.out
    assert "vllm: skipped (package not installed)" in captured.out


def test_build_workload_supports_small_max_lengths() -> None:
    args = bench.parse_args(
        [
            "--backend",
            "inferlib",
            "--num-seqs",
            "2",
            "--max-input-len",
            "8",
            "--max-output-len",
            "8",
        ]
    )

    prompts, completions = bench.build_workload(args)

    assert len(prompts) == 2
    assert all(1 <= len(prompt) <= 8 for prompt in prompts)
    assert all(1 <= limit <= 8 for limit in completions)
