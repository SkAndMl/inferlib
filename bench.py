from __future__ import annotations

import argparse
import os
import time

from dataclasses import dataclass
from random import randint, seed

from inferlib import LLM, SamplingParams


DEFAULT_MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


@dataclass(slots=True)
class BenchmarkResult:
    backend: str
    total_tokens: int
    elapsed_seconds: float

    @property
    def throughput(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_tokens / self.elapsed_seconds


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inferlib against vLLM.")
    parser.add_argument("--backend", choices=["inferlib", "vllm", "both"], default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-seqs", type=_positive_int, default=256)
    parser.add_argument("--max-input-len", type=_positive_int, default=1024)
    parser.add_argument("--max-output-len", type=_positive_int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=_positive_int, default=4096)
    return parser.parse_args(argv)


def build_workload(args: argparse.Namespace) -> tuple[list[list[int]], list[int]]:
    seed(args.seed)
    min_input_len = min(100, args.max_input_len)
    min_output_len = min(100, args.max_output_len)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    completion_token_limits = [
        randint(min_output_len, args.max_output_len) for _ in range(args.num_seqs)
    ]
    return prompt_token_ids, completion_token_limits


def print_status(message: str) -> None:
    print(message, flush=True)


def benchmark_inferlib(
    *,
    model: str,
    prompt_token_ids: list[list[int]],
    completion_token_limits: list[int],
    max_model_len: int,
) -> BenchmarkResult:
    print_status(f"inferlib: loading model {model}")
    llm = LLM(model, enforce_eager=False, max_model_len=max_model_len)
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
        for max_tokens in completion_token_limits
    ]
    print_status("inferlib: running warmup request")
    llm.generate(["Benchmark: "], SamplingParams())
    print_status(
        f"inferlib: running benchmark for {len(prompt_token_ids)} sequences "
        f"and {sum(completion_token_limits)} planned output tokens"
    )
    started_at = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed_seconds = time.perf_counter() - started_at
    llm.close()
    return BenchmarkResult(
        backend="inferlib",
        total_tokens=sum(completion_token_limits),
        elapsed_seconds=elapsed_seconds,
    )


def benchmark_vllm(
    *,
    model: str,
    prompt_token_ids: list[list[int]],
    completion_token_limits: list[int],
    max_model_len: int,
) -> BenchmarkResult | None:
    try:
        from vllm import LLM as VllmLLM
        from vllm import SamplingParams as VllmSamplingParams
    except ImportError:
        return None

    print_status(f"vllm: loading model {model}")
    llm = VllmLLM(model, enforce_eager=False, max_model_len=max_model_len)
    sampling_params = [
        VllmSamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
        for max_tokens in completion_token_limits
    ]
    print_status("vllm: running warmup request")
    llm.generate(["Benchmark: "], VllmSamplingParams())
    print_status(
        f"vllm: running benchmark for {len(prompt_token_ids)} sequences "
        f"and {sum(completion_token_limits)} planned output tokens"
    )
    started_at = time.perf_counter()
    llm.generate(
        [dict(prompt_token_ids=prompt) for prompt in prompt_token_ids],
        sampling_params,
        use_tqdm=False,
    )
    elapsed_seconds = time.perf_counter() - started_at
    return BenchmarkResult(
        backend="vllm",
        total_tokens=sum(completion_token_limits),
        elapsed_seconds=elapsed_seconds,
    )


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.backend}: Total {result.total_tokens}tok, "
        f"Time {result.elapsed_seconds:.2f}s, "
        f"Throughput {result.throughput:.2f}tok/s"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt_token_ids, completion_token_limits = build_workload(args)
    print_status(
        f"workload: backend={args.backend} model={args.model} num_seqs={args.num_seqs} "
        f"max_input_len={args.max_input_len} max_output_len={args.max_output_len} "
        f"max_model_len={args.max_model_len}"
    )

    if args.backend in {"inferlib", "both"}:
        print_result(
            benchmark_inferlib(
                model=args.model,
                prompt_token_ids=prompt_token_ids,
                completion_token_limits=completion_token_limits,
                max_model_len=args.max_model_len,
            )
        )

    if args.backend in {"vllm", "both"}:
        result = benchmark_vllm(
            model=args.model,
            prompt_token_ids=prompt_token_ids,
            completion_token_limits=completion_token_limits,
            max_model_len=args.max_model_len,
        )
        if result is None:
            print("vllm: skipped (package not installed)")
        else:
            print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
