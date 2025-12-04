import time
import csv
from typing import List, Dict

import numpy as np
import torch

from utils import load_model_and_tokenizer
from processor import PrePackProcessor
from profiling_time_and_memory import (
    prefill_with_prepacking,
    prefill_with_baseline,
    TTFT_with_prepacking,
    TTFT_with_baseline,
)


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
) -> float:
    """
    Run a batch and return latency (seconds).

    metric:
        - "prefill": measure only prefill time
        - "ttft": measure prefill + generate-first-token time (Time To First Token)
    """
    if method == "prepacking":
        if processor is None:
            processor = PrePackProcessor(tokenizer)
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
    max_tokens: int = None,
) -> float:
    """
    Static prepack：fixed wait window fixed_wait_window (seconds),
    return average per-input TTFT (seconds).

    TTFT here includes:
      - waiting time in the queue
      - batch prefill + generate-1-token time (via TTFT_* functions)
    
    Args:
        max_tokens: Maximum total tokens allowed in a batch. If None, no limit.
    """
    processor = PrePackProcessor(tokenizer)
    ttfts = []

    i = 0
    server_time = 0.0  # server current free time
    batch_id = 0

    n = len(records)
    while i < n:
        arrival = records[i]["timestamp"]
        # if the server is free and the next request is earlier, wait for the request to arrive
        if arrival > server_time:
            server_time = arrival

        # fixed wait window
        batch_start = server_time + fixed_wait_window

        # collect requests that arrive within the window
        batch_indices = []
        total_tokens = 0
        j = i
        while j < n and records[j]["timestamp"] <= batch_start:
            # Check max_tokens limit if specified
            if max_tokens is not None:
                # Tokenize the current request to count tokens
                request_tokens = len(tokenizer(records[j]["text"]).input_ids)
                if total_tokens + request_tokens > max_tokens:
                    # Adding this request would exceed max_tokens, stop collecting
                    break
                total_tokens += request_tokens
            batch_indices.append(j)
            j += 1

        if not batch_indices:
            # theoretically not possible, if it happens, move to the next request
            i += 1
            continue

        batch_texts = [records[k]["text"] for k in batch_indices]

        # print current batch information
        batch_id += 1
        token_info = f", tokens={total_tokens}" if max_tokens is not None else ""
        print(
            f"[STATIC] batch={batch_id}, "
            f"window={fixed_wait_window:.4f}s, "
            f"num_requests={len(batch_indices)}{token_info}"
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
        )

        batch_finish = batch_start + batch_latency

        # For each request in the batch, calculate TTFT = (batch_finish - arrival_time)
        for k in batch_indices:
            arrival_k = records[k]["timestamp"]
            ttfts.append(batch_finish - arrival_k)

        # update pointer and server time
        i = batch_indices[-1] + 1
        server_time = batch_finish

    return float(np.mean(ttfts))


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
    max_tokens: int = None,
) -> float:
    """
    AIMD prepack：wait window is dynamically controlled by AIMDWindowController,
    return average per-input TTFT (seconds).

    TTFT here includes:
      - waiting time in the queue
      - batch prefill + generate-1-token time (via TTFT_* functions)
    
    Args:
        max_tokens: Maximum total tokens allowed in a batch. If None, no limit.
    """
    processor = PrePackProcessor(tokenizer)
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

    while i < n:
        arrival = records[i]["timestamp"]
        if arrival > server_time:
            server_time = arrival

        wait_window = controller.wait_window
        batch_start = server_time + wait_window

        batch_indices = []
        total_tokens = 0
        j = i
        while j < n and records[j]["timestamp"] <= batch_start:
            # Check max_tokens limit if specified
            if max_tokens is not None:
                # Tokenize the current request to count tokens
                request_tokens = len(tokenizer(records[j]["text"]).input_ids)
                if total_tokens + request_tokens > max_tokens:
                    # Adding this request would exceed max_tokens, stop collecting
                    break
                total_tokens += request_tokens
            batch_indices.append(j)
            j += 1

        if not batch_indices:
            i += 1
            continue

        batch_texts = [records[k]["text"] for k in batch_indices]

        # print current batch information (using current window)
        batch_id += 1
        token_info = f", tokens={total_tokens}" if max_tokens is not None else ""
        print(
            f"[AIMD] batch={batch_id}, "
            f"wait_window={wait_window:.4f}s, "
            f"num_requests={len(batch_indices)}{token_info}"
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


if __name__ == "__main__":
    # 1. load timestamp + text data
    COMBINED_CSV_PATH = "../dataset/combined/mmlu_azure_ts.csv"
    records = load_records(COMBINED_CSV_PATH)
    print(f"Loaded {len(records)} records from {COMBINED_CSV_PATH}")

    # 2. load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(base_model="llama2.7b", loadbit=4)
    device = model.device

    # 3. TEST：static prepack
    static_wait = 0.1
    avg_ttft_static = simulate_static_wait(
        records,
        model,
        tokenizer,
        device,
        fixed_wait_window=static_wait,
        method="prepacking",
        max_tokens=2048,
    )
    print(f"[STATIC] fixed_wait={static_wait:.3f}s, avg per-input TTFT={avg_ttft_static:.4f}s")

    # 4. TEST：AIMD prepack
    avg_ttft_aimd = simulate_aimd_wait(
        records,
        model,
        tokenizer,
        device,
        init_wait=0.2,  # initial wait window
        min_wait=0.05,  # minimum wait window (as long as there is a request)
        max_wait=0.4,  # maximum wait window
        target_ttft=0.3,  # target per-input TTFT, adjust according to your needs
        method="prepacking",
        alpha=0.05,  # when latency < target, wait += alpha
        beta=0.5,  # when latency >= target, wait *= beta
        max_tokens=2048,
    )
    print(f"[AIMD] avg per-input TTFT={avg_ttft_aimd:.4f}s")
