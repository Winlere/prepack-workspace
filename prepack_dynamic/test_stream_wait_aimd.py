import time
import csv
from typing import List, Dict

import numpy as np
import torch
import sys

from utils import load_model_and_tokenizer, integer_program_packing, LAST_ILP_SOLVE_TIME
from processor import PrePackProcessor
from profiling_time_and_memory import (
    prefill_with_prepacking,
    prefill_with_baseline,
    TTFT_with_prepacking,
    TTFT_with_baseline,
)

COMBINED_CSV_PATH = "../dataset/realdata_downsample/code.sampled.0.csv"
# COMBINED_CSV_PATH = "../dataset/combined/mmlu_azure_ts_scaled_4.csv"


def load_records(csv_path: str) -> List[Dict]:
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp"])
            text = row["text"]
            records.append({"timestamp": t, "text": text})
            
    records.sort(key=lambda r: r["timestamp"]) # could be removed if the csv is already sorted by timestamp
    return records


def run_batch(
    sentences,
    model,
    tokenizer,
    device,
    method: str = "prepacking",
    processor: PrePackProcessor = None,
) -> float:
    """Deprecated: use run_batch_with_metric instead."""
    raise NotImplementedError("Use run_batch_with_metric instead.")


def run_batch_with_metric(
    sentences,
    model,
    tokenizer,
    device,
    method: str = "prepacking",
    metric: str = "prefill",
    processor: PrePackProcessor = None,
    use_ilp_packing: bool = False,
) -> float:
    """
    Run a batch and return latency (seconds).

    metric:
        - "prefill": measure only prefill time
        - "ttft": measure prefill + generate-first-token time (Time To First Token)
    """
    if method == "prepacking":
        if processor is None:
            # 默认使用 greedy（binpacking 库内部的 FFD），当 use_ilp_packing=True 时
            # 使用 utils.integer_program_packing（ILP 最优解）
            packing_fn = integer_program_packing if use_ilp_packing else None
            processor = PrePackProcessor(tokenizer, packing_fn=packing_fn)
        if metric == "prefill":
            fn = prefill_with_prepacking
        elif metric == "ttft":
            fn = TTFT_with_prepacking
        else:
            raise ValueError(f"Unknown metric: {metric}")
        fn_args = (sentences, model, tokenizer, device, processor)
    elif method == "baseline":
        if metric == "prefill":
            fn = prefill_with_baseline
        elif metric == "ttft":
            fn = TTFT_with_baseline
        else:
            raise ValueError(f"Unknown metric: {metric}")
        fn_args = (sentences, model, tokenizer, device, None)
    else:
        raise ValueError(f"Unknown method: {method}")

    torch.cuda.empty_cache()
    start = time.time()
    _ = fn(*fn_args)
    return time.time() - start


def simulate_static_wait(
    records: List[Dict],
    model,
    tokenizer,
    device,
    fixed_wait_window: float,
    method: str = "prepacking",
    max_requests_per_batch: int = 64,
    max_tokens: int = None,
    use_ilp_packing: bool = False,
) -> float:
    """
    Static prepack：fixed wait window fixed_wait_window (seconds),
    return average per-input TTFT (seconds).

    TTFT here includes:
      - waiting time in the queue
      - batch prefill + generate-1-token time (via TTFT_* functions)
    """
    packing_fn = integer_program_packing if use_ilp_packing else None
    processor = PrePackProcessor(tokenizer, packing_fn=packing_fn)
    ttfts = []

    i = 0
    server_time = 0.0  # server current free time
    batch_id = 0
    first_arrival = None  # Track first request arrival time
    total_tokens_all_batches = 0  # Track total tokens processed across all batches

    n = len(records)
    while i < n:
        arrival = records[i]["timestamp"]
        # Track first arrival time
        if first_arrival is None:
            first_arrival = arrival
        # if the server is free and the next request is earlier, wait for the request to arrive
        if arrival > server_time:
            server_time = arrival

        # fixed wait window
        batch_start = server_time + fixed_wait_window

        batch_indices = []
        token_count = 0
        j = i
        while (
            j < n
            and records[j]["timestamp"] <= batch_start
            and len(batch_indices) < max_requests_per_batch
        ):
            token_count_j = len(tokenizer(records[j]["text"]).input_ids)
            if max_tokens is not None and token_count + token_count_j > max_tokens:
                break
            batch_indices.append(j)
            token_count += token_count_j
            j += 1

        if not batch_indices:
            # theoretically not possible, if it happens, move to the next request
            i += 1
            continue

        batch_texts = [records[k]["text"] for k in batch_indices]

        total_tokens = token_count
        total_tokens_all_batches += total_tokens

        # print current batch information
        batch_id += 1
        print(
            f"[STATIC] batch={batch_id}, "
            f"window={fixed_wait_window:.4f}s, "
            f"num_requests={len(batch_indices)}"
        )

        # Run one batch with TTFT metric, batch latency includes (prefill + generate first token)
        batch_latency = run_batch_with_metric(
            batch_texts,
            model,
            tokenizer,
            device,
            method=method,
            metric="ttft",
            processor=processor,
            use_ilp_packing=use_ilp_packing,
        )

        batch_finish = batch_start + batch_latency

        # For each request in the batch, calculate TTFT = (batch_finish - arrival_time)
        for k in batch_indices:
            arrival_k = records[k]["timestamp"]
            ttfts.append(batch_finish - arrival_k)

        # update pointer and server time
        i = batch_indices[-1] + 1
        server_time = batch_finish

    avg_ttft = float(np.mean(ttfts))
    total_time = server_time - first_arrival if first_arrival is not None else 0.0
    throughput = total_tokens_all_batches / total_time if total_time > 0 else 0.0
    print(f"[STATIC] Throughput: {throughput:.2f} tokens/second")
    
    return avg_ttft


class AIMDWindowController:
    """
    AIMD controller for controlling wait window (seconds):
      - when latency < target, wait += alpha
      - when latency >= target, wait *= beta
      - clamp to [min_wait, max_wait]
    """

    def __init__(self, init_wait, min_wait, max_wait, target_latency, alpha=0.01, beta=0.5):
        self.init_wait = float(init_wait)
        self.min_wait = float(min_wait)
        self.max_wait = float(max_wait)
        self.target_latency = float(target_latency)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.wait_window = max(self.min_wait, min(self.init_wait, self.max_wait))

    def update(self, last_latency: float) -> float:
        if last_latency < self.target_latency:
            self.wait_window += self.alpha
        else:
            self.wait_window *= self.beta

        self.wait_window = max(self.min_wait, min(self.wait_window, self.max_wait))
        return self.wait_window


class AIMDSizeController:
    """
    AIMD controller on batch size (number of requests per batch):
      - if backlog_after_batch > current_size: size += alpha  (additive increase)
      - else:                                    size *= beta  (multiplicative decrease)
      - clamp to [min_size, max_size]
    """

    def __init__(self, init_size, min_size, max_size, alpha=1.0, beta=0.5):
        self.init_size = int(init_size)
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.batch_size = max(self.min_size, min(self.init_size, self.max_size))

    def update(self, backlog_after_batch: int) -> int:
        if backlog_after_batch > self.batch_size:
            self.batch_size += self.alpha
        else:
            self.batch_size *= self.beta

        self.batch_size = int(max(self.min_size, min(self.batch_size, self.max_size)))
        if self.batch_size < 1:
            self.batch_size = 1
        return self.batch_size


def simulate_size_aimd_wait_advanced(
    records: List[Dict],
    model,
    tokenizer,
    device,
    init_size: int,
    min_size: int,
    max_size: int,
    method: str = "prepacking",
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    tau_low: float = 0.25,
    tau_high: float = 0.35,
    max_requests_per_batch: int = 64,
    backlog_hi_factor: float = 2.0,
    backlog_lo_factor: float = 0.5,
    max_tokens: int = None,
) -> float:
    """
    Advanced size-based AIMD batching with smoothed TTFT_p95, hysteresis band, and backlog-aware logic.

    - Maintain desired batch size N_t in [min_size, max_size].
    - For each batch (epoch t):
        * Form a batch using at most N_t requests (plus hard cap max_requests_per_batch).
        * Compute per-input TTFTs and TTFT_p95 for this batch.
        * Update smoothed signal S_t = EWMA_gamma(TTFT_p95).
        * Compute backlog_after at batch_finish.
        * Update N_{t+1}:
              if S_t <= tau_low and backlog_after > backlog_hi_factor * N_t:
                    N_{t+1} = min(N_max, N_t + alpha)          (AI, high utilization)
              elif S_t >= tau_high and backlog_after < backlog_lo_factor * N_t:
                    N_{t+1} = max(N_min, ceil(beta * N_t))     (MD, low backlog)
              else:
                    N_{t+1} = N_t                              (stable / conflicting signals)
    """
    processor = PrePackProcessor(tokenizer, packing_fn=integer_program_packing)

    desired_size = max(min_size, min(init_size, max_size))
    S_t = None  # smoothed TTFT signal

    ttfts = []
    i = 0
    server_time = 0.0
    n = len(records)
    batch_id = 0

    while i < n:
        # Ensure server_time is at least the arrival of the next unserved request
        arrival = records[i]["timestamp"]
        if arrival > server_time:
            server_time = arrival

        # Backlog at this time (capped)
        queued_indices = []
        j = i
        while (
            j < n
            and records[j]["timestamp"] <= server_time
            and len(queued_indices) < max_requests_per_batch
        ):
            queued_indices.append(j)
            j += 1

        if not queued_indices:
            i += 1
            continue

        actual_batch_size = max(1, min(len(queued_indices), desired_size, max_requests_per_batch))
        batch_indices = []
        token_count = 0
        for k in queued_indices[:actual_batch_size]:
            token_count_k = len(tokenizer(records[k]["text"]).input_ids)
            if max_tokens is not None and token_count + token_count_k > max_tokens:
                break
            batch_indices.append(k)
            token_count += token_count_k
        batch_texts = [records[k]["text"] for k in batch_indices]

        batch_id += 1
        print(
            f"[SIZE-AIMD-ADV] batch={batch_id}, "
            f"desired_size={desired_size}, "
            f"backlog_before={len(queued_indices)}, "
            f"actual_batch_size={len(batch_indices)}"
        )

        batch_start = server_time
        batch_latency = run_batch_with_metric(
            batch_texts,
            model,
            tokenizer,
            device,
            method=method,
            metric="ttft",
            processor=processor,
            use_ilp_packing=True,
        )
        batch_finish = batch_start + batch_latency

        # Per-request TTFTs for this batch
        batch_ttfts = []
        for k in batch_indices:
            arrival_k = records[k]["timestamp"]
            ttft_k = batch_finish - arrival_k
            ttfts.append(ttft_k)
            batch_ttfts.append(ttft_k)

        if not batch_ttfts:
            i = batch_indices[-1] + 1
            server_time = batch_finish
            continue

        # Compute TTFT_p95 for this batch
        ttft_p95 = float(np.percentile(batch_ttfts, 95))

        # Update smoothed signal S_t (EWMA)
        if S_t is None:
            S_t = ttft_p95
        else:
            S_t = (1.0 - gamma) * S_t + gamma * ttft_p95

        # Compute backlog_after at batch_finish
        backlog_after = 0
        k = batch_indices[-1] + 1
        while k < n and records[k]["timestamp"] <= batch_finish:
            backlog_after += 1
            k += 1

        # Backlog-aware hysteresis-based AIMD update on desired_size
        old_size = desired_size
        if S_t <= tau_low and backlog_after > backlog_hi_factor * desired_size:
            # Latency is good and backlog is high: be more aggressive
            desired_size = min(max_size, desired_size + alpha)
        elif S_t >= tau_high and backlog_after < backlog_lo_factor * desired_size:
            # Latency is bad and backlog is low: shrink batch size
            desired_size = max(min_size, int(np.ceil(beta * desired_size)))
        # else: conflicting signals or within band -> keep desired_size unchanged

        desired_size = int(max(min_size, min(desired_size, max_size)))
        if desired_size < 1:
            desired_size = 1

        print(
            f"[SIZE-AIMD-ADV] TTFT_p95={ttft_p95:.4f}, S_t={S_t:.4f}, "
            f"backlog_after={backlog_after}, "
            f"updated_desired_size={desired_size} (was {old_size})"
        )

        i = batch_indices[-1] + 1
        server_time = batch_finish

    return float(np.mean(ttfts))

def simulate_aimd_wait(
    records: List[Dict],
    model,
    tokenizer,
    device,
    init_wait: float,
    min_wait: float,
    max_wait: float,
    target_ttft: float,
    method: str = "prepacking",
    alpha: float = 0.01,
    beta: float = 0.5,
    max_requests_per_batch: int = 64,
    max_tokens: int = None,
    use_ilp_packing: bool = False,
) -> float:
    """
    AIMD prepack：wait window is dynamically controlled by AIMDWindowController,
    return average per-input TTFT (seconds).

    TTFT here includes:
      - waiting time in the queue
      - batch prefill + generate-1-token time (via TTFT_* functions)
    """
    packing_fn = integer_program_packing if use_ilp_packing else None
    processor = PrePackProcessor(tokenizer, packing_fn=packing_fn)
    controller = AIMDWindowController(
        init_wait=init_wait,
        min_wait=min_wait,
        max_wait=max_wait,
        target_latency=target_ttft,
        alpha=alpha,
        beta=beta,
    )

    ttfts = []

    i = 0
    server_time = 0.0
    n = len(records)
    batch_id = 0
    first_arrival = None  # Track first request arrival time
    total_tokens_all_batches = 0  # Track total tokens processed across all batches

    while i < n:
        arrival = records[i]["timestamp"]
        # Track first arrival time
        if first_arrival is None:
            first_arrival = arrival
        if arrival > server_time:
            server_time = arrival

        wait_window = controller.wait_window
        batch_start = server_time + wait_window

        batch_indices = []
        token_count = 0
        j = i
        while (
            j < n
            and records[j]["timestamp"] <= batch_start
            and len(batch_indices) < max_requests_per_batch
        ):
            token_count_j = len(tokenizer(records[j]["text"]).input_ids)
            if max_tokens is not None and token_count + token_count_j > max_tokens:
                break
            batch_indices.append(j)
            token_count += token_count_j
            j += 1

        if not batch_indices:
            i += 1
            continue

        batch_texts = [records[k]["text"] for k in batch_indices]

        total_tokens = token_count
        total_tokens_all_batches += total_tokens

        # print current batch information (using current window)
        batch_id += 1
        print(
            f"[AIMD] batch={batch_id}, "
            f"wait_window={wait_window:.4f}s, "
            f"num_requests={len(batch_indices)}"
        )

        # Run one batch with TTFT metric, batch latency includes (prefill + generate first token)
        batch_latency = run_batch_with_metric(
            batch_texts,
            model,
            tokenizer,
            device,
            method=method,
            metric="ttft",
            processor=processor,
            use_ilp_packing=use_ilp_packing,
        )

        batch_finish = batch_start + batch_latency

        # average TTFT of this batch, as feedback to AIMD
        batch_ttfts = [batch_finish - records[k]["timestamp"] for k in batch_indices]
        avg_batch_ttft = float(np.mean(batch_ttfts))
        ttfts.extend(batch_ttfts)

        controller.update(avg_batch_ttft)

        i = batch_indices[-1] + 1
        server_time = batch_finish

    return float(np.mean(ttfts))


def simulate_size_based_wait(
    records: List[Dict],
    model,
    tokenizer,
    device,
    target_batch_size: int = 10,
    method: str = "prepacking",
    max_tokens: int = None,
    use_ilp_packing: bool = False,
) -> float:
    """
    Size-based batching: wait until there are `target_batch_size` requests,
    then immediately run a batch with those requests (no extra time-based wait).

    Semantics:
      - For each batch, we wait until the LAST request in this batch arrives,
        then immediately start the batch.
      - TTFT includes:
          * waiting time until the last request of the batch arrives
          * batch prefill + generate-1-token time (via TTFT_* functions)
    """
    packing_fn = integer_program_packing if use_ilp_packing else None
    processor = PrePackProcessor(tokenizer, packing_fn=packing_fn)
    ttfts = []

    i = 0
    server_time = 0.0
    n = len(records)
    batch_id = 0

    while i < n:
        batch_indices = []
        token_count = 0
        j = i
        while (
            j < n
            and len(batch_indices) < target_batch_size
        ):
            token_count_j = len(tokenizer(records[j]["text"]).input_ids)
            if max_tokens is not None and token_count + token_count_j > max_tokens:
                break
            batch_indices.append(j)
            token_count += token_count_j
            j += 1
        if not batch_indices:
            break

        batch_id += 1
        batch_texts = [records[k]["text"] for k in batch_indices]

        # The batch cannot start before the last request in this batch arrives
        arrival_last = records[batch_indices[-1]]["timestamp"]
        if arrival_last > server_time:
            server_time = arrival_last
        batch_start = server_time

        print(
            f"[SIZE-{target_batch_size}] batch={batch_id}, "
            f"num_requests={len(batch_indices)}"
        )

        batch_latency = run_batch_with_metric(
            batch_texts,
            model,
            tokenizer,
            device,
            method=method,
            metric="ttft",
            processor=processor,
            use_ilp_packing=use_ilp_packing,
        )
        batch_finish = batch_start + batch_latency

        for k in batch_indices:
            arrival_k = records[k]["timestamp"]
            ttfts.append(batch_finish - arrival_k)

        i = batch_indices[-1] + 1
        server_time = batch_finish

    return float(np.mean(ttfts))


def simulate_size_aimd_wait(
    records: List[Dict],
    model,
    tokenizer,
    device,
    init_size: int,
    min_size: int,
    max_size: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    method: str = "prepacking",
    max_requests_per_batch: int = 64,
    max_tokens: int = None,
    use_ilp_packing: bool = False,
) -> float:
    """
    Size-based AIMD batching:

    - Controller maintains a desired batch size (number of requests), S.
    - At each batch:
        * Let backlog_before = queued requests at server_time (capped by max_requests_per_batch).
        * Actual batch size = min(backlog_before, S, max_requests_per_batch), if backlog_before > 0.
        * After running the batch, compute backlog_after at batch_finish.
        * If backlog_after > S: S += alpha      (additive increase)
          else:               S *= beta        (multiplicative decrease)
      TTFT includes queue waiting time + batch TTFT (prefill + first token).
    """
    packing_fn = integer_program_packing if use_ilp_packing else None
    processor = PrePackProcessor(tokenizer, packing_fn=packing_fn)
    controller = AIMDSizeController(
        init_size=init_size,
        min_size=min_size,
        max_size=max_size,
        alpha=alpha,
        beta=beta,
    )

    ttfts = []
    i = 0
    server_time = 0.0
    n = len(records)
    batch_id = 0

    while i < n:
        # Ensure server_time is at least the arrival of the next unserved request
        arrival = records[i]["timestamp"]
        if arrival > server_time:
            server_time = arrival

        # Compute backlog before starting this batch
        queued_indices = []
        j = i
        while (
            j < n
            and records[j]["timestamp"] <= server_time
            and len(queued_indices) < max_requests_per_batch
        ):
            queued_indices.append(j)
            j += 1

        if not queued_indices:
            # No queued requests at this time, move to next
            i += 1
            continue

        desired_size = controller.batch_size
        actual_batch_size = max(1, min(len(queued_indices), desired_size, max_requests_per_batch))

        batch_indices = []
        token_count = 0
        for k in queued_indices[:actual_batch_size]:
            token_count_k = len(tokenizer(records[k]["text"]).input_ids)
            if max_tokens is not None and token_count + token_count_k > max_tokens:
                break
            batch_indices.append(k)
            token_count += token_count_k
        batch_texts = [records[k]["text"] for k in batch_indices]

        batch_id += 1
        print(
            f"[SIZE-AIMD] batch={batch_id}, "
            f"desired_size={desired_size}, "
            f"backlog_before={len(queued_indices)}, "
            f"actual_batch_size={len(batch_indices)}"
        )

        batch_start = server_time

        batch_latency = run_batch_with_metric(
            batch_texts,
            model,
            tokenizer,
            device,
            method=method,
            metric="ttft",
            processor=processor,
            use_ilp_packing=use_ilp_packing,
        )
        batch_finish = batch_start + batch_latency

        # Per-request TTFT
        for k in batch_indices:
            arrival_k = records[k]["timestamp"]
            ttfts.append(batch_finish - arrival_k)

        # Compute backlog after batch_finish (unserved + already arrived)
        backlog_after = 0
        k = batch_indices[-1] + 1
        while k < n and records[k]["timestamp"] <= batch_finish:
            backlog_after += 1
            k += 1

        controller.update(backlog_after)

        i = batch_indices[-1] + 1
        server_time = batch_finish

    return float(np.mean(ttfts))

def simulate_first_wait_then_zero_wait(
    records: List[Dict],
    model,
    tokenizer,
    device,
    first_wait_window: float,
    method: str = "prepacking",
    max_requests_per_batch: int = 64,
    max_tokens: int = None,
    use_ilp_packing: bool = False,
    overlap_ilp: bool = False,
) -> float:
    """
    Hybrid policy:
      - First batch: use a time-based wait window `first_wait_window` seconds.
      - Subsequent batches: wait_window = 0, i.e.,
          * if there are queued requests when previous batch finishes, run immediately;
          * otherwise, wait until next arrival, then run immediately (no extra delay).

    当 `use_ilp_packing=True` 且 `overlap_ilp=True` 时，我们采用“理想 overlap”近似：
      - batch 的组成方式完全保持不变（仍按 first_wait_window / zero_wait 规则选取请求）；
      - 仍然真实运行一次 ILP+GPU 来获得 batch_latency 和 LAST_ILP_SOLVE_TIME；
      - 在仿真时间线上，将本批的有效延迟视为 (batch_latency - LAST_ILP_SOLVE_TIME)，
        相当于假设 ILP 计算完全被上一批 GPU 运行时间隐藏。
    """
    packing_fn = integer_program_packing if use_ilp_packing else None
    processor = PrePackProcessor(tokenizer, packing_fn=packing_fn)
    ttfts = []

    i = 0
    server_time = 0.0
    n = len(records)
    batch_id = 0
    first_batch = True

    while i < n:
        arrival = records[i]["timestamp"]
        if arrival > server_time:
            server_time = arrival

        wait_window = first_wait_window if first_batch else 0.0
        batch_start = server_time + wait_window

        batch_indices = []
        token_count = 0
        j = i
        while (
            j < n
            and records[j]["timestamp"] <= batch_start
            and len(batch_indices) < max_requests_per_batch
        ):
            token_count_j = len(tokenizer(records[j]["text"]).input_ids)
            if max_tokens is not None and token_count + token_count_j > max_tokens:
                break
            batch_indices.append(j)
            token_count += token_count_j
            j += 1

        if not batch_indices:
            i += 1
            first_batch = False
            continue

        batch_texts = [records[k]["text"] for k in batch_indices]

        batch_id += 1
        print(
            f"[FIRST-THEN-0] batch={batch_id}, "
            f"wait_window={wait_window:.4f}s, "
            f"num_requests={len(batch_indices)}"
        )

        batch_latency = run_batch_with_metric(
            batch_texts,
            model,
            tokenizer,
            device,
            method=method,
            metric="ttft",
            processor=processor,
            use_ilp_packing=use_ilp_packing,
        )

        # 理想 overlap：如果使用 ILP 且允许 overlap，就从 batch_latency 中扣掉 ILP 时间
        effective_latency = batch_latency
        if use_ilp_packing and overlap_ilp:
            ilp_time = LAST_ILP_SOLVE_TIME
            effective_latency = max(0.0, batch_latency - ilp_time)
            print(
                f"[FIRST-THEN-0-IDEAL-OVERLAP] batch={batch_id}, "
                f"ilp_time={ilp_time:.6f}s, "
                f"gpu_time={effective_latency:.6f}s"
            )

        batch_finish = batch_start + effective_latency

        for k in batch_indices:
            arrival_k = records[k]["timestamp"]
            ttfts.append(batch_finish - arrival_k)

        i = batch_indices[-1] + 1
        server_time = batch_finish
        first_batch = False

    return float(np.mean(ttfts))


if __name__ == "__main__":
    # 1. load timestamp + text data
    import argparse
    parser = argparse.ArgumentParser("")
    parser.add_argument("csv_file")

    args = parser.parse_args()
    COMBINED_CSV_PATH = args.csv_file

    records = load_records(COMBINED_CSV_PATH)
    print(f"Loaded {len(records)} records from {COMBINED_CSV_PATH}")

    # 2. load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(base_model="llama1b", loadbit=4)
    device = model.device

    max_tokens = 1024 * 20

    # # 3. TEST：static prepack (with prepacking)
    static_wait = 0.2
    max_requests_per_batch = 64
    # avg_ttft_static_prepack = simulate_static_wait(
    #     records,
    #     model,
    #     tokenizer,
    #     device,
    #     fixed_wait_window=static_wait,
    #     method="prepacking",
    #     max_requests_per_batch=max_requests_per_batch,
    # )
    # print(
    #     f"[STATIC-PREPACK] "
    #     f"avg per-input TTFT={avg_ttft_static_prepack:.4f}s"
    # )

    # # 3b. TEST：static baseline (no prepacking, padding-based batching)
    # avg_ttft_static_baseline = simulate_static_wait(
    #     records,
    #     model,
    #     tokenizer,
    #     device,
    #     fixed_wait_window=static_wait,
    #     method="baseline",
    #     max_requests_per_batch=max_requests_per_batch,
    # )
    # print(
    #     f"[STATIC-BASELINE] "
    #     f"avg per-input TTFT={avg_ttft_static_baseline:.4f}s"
    # )

    # # 4. TEST：AIMD prepack
    # avg_ttft_aimd_prepack = simulate_aimd_wait(
    #     records,
    #     model,
    #     tokenizer,
    #     device,
    #     init_wait=0.2,   # initial wait window
    #     min_wait=0.1,     # minimum wait window (as long as there is a request)
    #     max_wait=0.4,     # maximum wait window
    #     target_ttft=0.3,  # target per-input TTFT, adjust according to your needs
    #     method="prepacking",
    #     alpha=0.05,       # when latency < target, wait += alpha
    #     beta=0.5,         # when latency >= target, wait *= beta
    #     max_requests_per_batch=max_requests_per_batch,
    # )
    # print(f"[AIMD-PREPACK] avg per-input TTFT={avg_ttft_aimd_prepack:.4f}s")

    # # 4b. TEST：AIMD baseline (no prepacking)
    # avg_ttft_aimd_baseline = simulate_aimd_wait(
    #     records,
    #     model,
    #     tokenizer,
    #     device,
    #     init_wait=0.2,
    #     min_wait=0.1,
    #     max_wait=0.4,
    #     target_ttft=0.3,
    #     method="baseline",
    #     alpha=0.05,
    #     beta=0.5,
    #     max_requests_per_batch=max_requests_per_batch,
    # )
    # print(f"[AIMD-BASELINE] avg per-input TTFT={avg_ttft_aimd_baseline:.4f}s")

    # 5. TEST：size-based batching (wait until 10 requests, then run)
    # avg_ttft_size10_prepack = simulate_size_based_wait(
    #     records,
    #     model,
    #     tokenizer,
    #     device,
    #     target_batch_size=10,
    #     method="prepacking",
    # )
    # print(f"[SIZE-10-PREPACK] avg per-input TTFT={avg_ttft_size10_prepack:.4f}s")
    # avg_ttft_first_then_zero_baseline = simulate_first_wait_then_zero_wait(
    #     records,
    #     model,
    #     tokenizer,
    #     device,
    #     first_wait_window=0.2,
    #     method="baseline",
    #     max_requests_per_batch=max_requests_per_batch,
    #     max_tokens=max_tokens,
    # )
    # print(f"[FIRST-THEN-0-BASELINE] avg per-input TTFT={avg_ttft_first_then_zero_baseline:.4f}s")
    # # 6. TEST：first batch wait=0.2s, then wait_window=0s
    avg_ttft_first_then_zero = simulate_first_wait_then_zero_wait(
        records,
        model,
        tokenizer,
        device,
        first_wait_window=0.2,
        method="prepacking",
        max_requests_per_batch=max_requests_per_batch,
        max_tokens=max_tokens,
        use_ilp_packing=False, # False for greedy, True for ip/dp
        overlap_ilp=False,
    )
    print(f"[FIRST-THEN-0-PREPACK] avg per-input TTFT={avg_ttft_first_then_zero:.4f}s")

    sys.stdout.flush()

    # 7. TEST：size-based AIMD batching (dynamic batch size by count)
    avg_ttft_size_aimd_prepack = simulate_size_aimd_wait(
        records,
        model,
        tokenizer,
        device,
        init_size=4,     # initial desired batch size
        min_size=1,
        max_size=16,
        method="prepacking",
        alpha=10,        # additive increase step
        beta=0.5,         # multiplicative decrease factor
        max_requests_per_batch=128,
        max_tokens=max_tokens,
        use_ilp_packing=False, # False for greedy, True for ip/dp
    )
    print(f"[SIZE-AIMD-PREPACK] avg per-input TTFT={avg_ttft_size_aimd_prepack:.4f}s")

    sys.stdout.flush()

    avg_ttft_size_aimd_prepack_dp = simulate_size_aimd_wait(
        records,
        model,
        tokenizer,
        device,
        init_size=4,     # initial desired batch size
        min_size=1,
        max_size=16,
        method="prepacking",
        alpha=10,        # additive increase step
        beta=0.5,         # multiplicative decrease factor
        max_requests_per_batch=128,
        max_tokens=max_tokens,
        use_ilp_packing=True, # False for greedy, True for ip/dp
    )
    print(f"[SIZE-AIMD-PREPACK-DP] avg per-input TTFT={avg_ttft_size_aimd_prepack_dp:.4f}s")

    # # 8. TEST：advanced size-based AIMD batching (EWMA + hysteresis on TTFT_p95) (haven't tested)
    # avg_ttft_size_aimd_adv_prepack = simulate_size_aimd_wait_advanced(
    #     records,
    #     model,
    #     tokenizer,
    #     device,
    #     init_size=64,
    #     min_size=4,
    #     max_size=128,
    #     method="prepacking",
    #     alpha=10,
    #     beta=0.5,
    #     gamma=0.1,
    #     tau_low=0.25,
    #     tau_high=0.35,
    #     max_requests_per_batch=128,
    #     max_tokens=max_tokens,
    # )
    # print(f"[SIZE-AIMD-ADV-PREPACK] avg per-input TTFT={avg_ttft_size_aimd_adv_prepack:.4f}s")