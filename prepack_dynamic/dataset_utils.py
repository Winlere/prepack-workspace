from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import random
import torch
from utils import left_pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from transformers.trainer_utils import set_seed
import csv
import os
from datetime import datetime

SEED = 41
set_seed(SEED)


class HFDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def load_and_prepare_data(
    dataset_name: str,
    config_name: str,
    split: str,
    tokenizer,
    sample_size: int = 1000,
    max_length: int = None,
):
    dataset = load_dataset(dataset_name, config_name, split=split)
    if dataset_name == "wikitext":
        texts = dataset["text"]
    elif "mmlu" in dataset_name:
        texts = dataset["question"]
    elif "rlhf" in dataset_name:
        texts = dataset["chosen"]
    elif "alpaca" in dataset_name:
        texts = dataset["text"]
    elif "samsum" in dataset_name:
        texts = dataset["dialogue"]
    print(len(texts), "texts loaded", sample_size)
    texts = [text for text in texts if len(text.split()) > 1]
    texts = random.sample(texts, sample_size)

    if max_length:
        tokenized_texts = [
            tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt") for text in texts
        ]
        texts = [
            tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True) for tokens in tokenized_texts
        ]
    return texts


def load_and_evaluate_dataset(dataset: str, tokenizer):
    dataset_config = {
        "mmlu": ("cais/mmlu", "all", None, 1000, "test"),
        "wikitext512": ("wikitext", "wikitext-2-raw-v1", 512, 1000, "test"),
        "wikitext256": ("wikitext", "wikitext-2-raw-v1", 256, 1000, "test"),
        "rlhf": ("Anthropic/hh-rlhf", "default", None, 1000, "train"),
        "alpaca": ("tatsu-lab/alpaca", "default", None, 1000, "train"),
        "samsum": ("samsum", "samsum", None, 1000, "train"),
    }

    if dataset not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataset_name, config_name, max_length, sample_size, split = dataset_config[dataset]

    # Load dataset
    texts = load_and_prepare_data(
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        tokenizer=tokenizer,
        sample_size=sample_size,
        max_length=max_length,
    )

    # Evaluate sentences from the dataset
    print(f"Evaluating {len(texts)} sentences from the dataset {dataset_name}")
    lengths = [len(tokenizer(text).input_ids) for text in texts]
    print("Mean length:", np.mean(lengths))
    print("Max length:", np.max(lengths))
    print("Min length:", np.min(lengths))

    # Plotting the length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color="skyblue", edgecolor="black")
    plt.title(f"Length Distribution of Texts in {dataset_name}")
    plt.xlabel("Length of Texts in Tokens")
    plt.ylabel("Frequency")
    plt.savefig(f"length_distribution_{dataset}.png")
    plt.close()

    return texts


def sample_batches(texts, batch_size):
    random.shuffle(texts)
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def sample_batches_deterministic(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def sample_batches_by_length(texts, batch_size):
    # Sort texts by their length
    texts_sorted = sorted(texts, key=lambda x: len(x))
    # Yield batches of similar length
    for i in range(0, len(texts_sorted), batch_size):
        yield texts_sorted[i : i + batch_size]


def sample_packed_dataset(dataset):
    # Assumption is dataset is already shuffled and batched
    for i in range(len(dataset)):
        yield dataset[i]


def unpack_kv(packed_outputs, restart_dict, original_ids, device):

    batch_size = sum(map(len, restart_dict)) - len(restart_dict)

    dim1, dim2 = len(packed_outputs), len(packed_outputs[0])

    save_cache = [[None for _ in range(dim2)] for _ in range(dim1)]
    batch_length = [len(ids) - 1 for ids in original_ids]
    compute = True

    attention_masks = torch.zeros((batch_size, max(batch_length) + 1), dtype=torch.int, device=device)
    final_tokens = torch.empty(batch_size, dtype=torch.int, device=device)

    for j in range(dim1):  # layer
        for k in range(dim2):  # k, v
            batch_cache = np.empty(batch_size, dtype=object)

            for b, batch in enumerate(restart_dict):
                batch_indices = list(batch.keys())
                for i in range(len(batch) - 1):
                    c = packed_outputs[j][k][b, :, batch_indices[i] : batch_indices[i + 1], :].permute(
                        1, 0, 2
                    )
                    original_index = restart_dict[b][batch_indices[i + 1]]
                    batch_cache[original_index] = c
                    if compute:
                        prompt = original_ids[batch[batch_indices[i + 1]]]
                        final_tokens[original_index] = prompt[-1]
                        attention_masks[original_index, -(batch_length[original_index]) - 1 :] = 1
            compute = False
            padded = left_pad_sequence(batch_cache, batch_first=True, padding_value=0).permute(0, 2, 1, 3)
            save_cache[j][k] = padded

    return save_cache, final_tokens.unsqueeze(dim=-1), attention_masks


class PackedDataset(Dataset):
    def __init__(self, new_tokens, new_positions, new_mask, restart_indices, original_ids, batch_size):
        self.tensor_tuple = (new_tokens, new_positions, new_mask)
        self.restart_indices = restart_indices
        self.original_ids = original_ids
        self.initial_processing(batch_size)

    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, idx):
        batch_idx = self.batch_indices[idx]
        tensors = tuple(tensor[batch_idx] for tensor in self.tensor_tuple)
        restart_idx = self.restart_indices[idx]
        original_id = self.original_ids[idx]
        return tensors + (restart_idx,) + (original_id,)

    def initial_processing(self, batch_size):

        num_samples = len(self.tensor_tuple[0])
        indices = list(range(num_samples))
        random.shuffle(indices)
        batch_indices = [indices[i : i + batch_size] for i in range(0, num_samples, batch_size)]

        # Purpose of below code is that with batching, restart indices and original ids must be reformatted
        # for consistency with existing unpack_kv function
        batched_restart_indices = []
        for idx in batch_indices:
            batch_restart_indices = list(map(self.restart_indices.__getitem__, idx))
            batched_restart_indices.append(batch_restart_indices)

        new_original_ids = []
        for idx in range(len(batch_indices)):
            original_id = []
            i = 0
            restart_idx = batched_restart_indices[idx]
            for d in restart_idx:
                for key in d:
                    value = d[key]
                    if value != -1:
                        original_id.append(self.original_ids[value])
                        d[key] = i
                        i += 1
            new_original_ids.append(original_id)

        self.original_ids = new_original_ids
        self.restart_indices = batched_restart_indices
        self.batch_indices = batch_indices


def load_mmlu_texts(tokenizer, sample_size: int = 1000):
    """
    Convenience wrapper to load only MMLU texts without plotting or statistics.
    This mirrors the configuration used in `load_and_evaluate_dataset`.
    """
    dataset_name = "cais/mmlu"
    config_name = "all"
    max_length = None
    split = "test"

    texts = load_and_prepare_data(
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        tokenizer=tokenizer,
        sample_size=sample_size,
        max_length=max_length,
    )
    return texts


def load_azure_trace_timestamps(
    csv_path: str,
    num_samples: int,
):
    """
    Load the first `num_samples` records from the Azure trace CSV and extract timestamps
    and token statistics.

    Returns:
        timestamps: list[float] - relative seconds from the first request
        context_tokens: list[int | None]
        generated_tokens: list[int | None]
    """
    timestamps = []
    context_tokens = []
    generated_tokens = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("TIMESTAMP")
            if ts_str is None:
                continue
            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
            except ValueError:
                # Skip rows with malformed timestamps
                continue

            ctx_raw = row.get("ContextTokens")
            gen_raw = row.get("GeneratedTokens")
            try:
                ctx_val = int(ctx_raw) if ctx_raw is not None else None
            except (TypeError, ValueError):
                ctx_val = None
            try:
                gen_val = int(gen_raw) if gen_raw is not None else None
            except (TypeError, ValueError):
                gen_val = None

            timestamps.append(ts)
            context_tokens.append(ctx_val)
            generated_tokens.append(gen_val)

            if len(timestamps) >= num_samples:
                break

    if not timestamps:
        return [], [], []

    # Normalize timestamps to start from 0 for easier simulation
    t0 = timestamps[0]
    timestamps = [t - t0 for t in timestamps]
    return timestamps, context_tokens, generated_tokens


def build_mmlu_azure_dataset(
    tokenizer,
    trace_csv_path: str,
    sample_size: int = 1000,
    save_path: str = None,
):
    """
    Build a synthetic dataset that pairs MMLU prompts with arrival timestamps
    taken from an Azure trace CSV.

    Each record has:
        - timestamp: float (relative seconds)
        - text: str (MMLU question)

    If `save_path` is provided, the combined dataset is also written to a CSV file.
    """
    # 1. Load MMLU texts
    mmlu_texts = load_mmlu_texts(tokenizer, sample_size=sample_size)

    # 2. Load timestamps and token stats from Azure trace
    timestamps, ctx_tokens, gen_tokens = load_azure_trace_timestamps(
        trace_csv_path,
        num_samples=len(mmlu_texts),
    )

    n = min(len(mmlu_texts), len(timestamps))
    mmlu_texts = mmlu_texts[:n]
    timestamps = timestamps[:n]
    ctx_tokens = ctx_tokens[:n]
    gen_tokens = gen_tokens[:n]

    records = []
    for i in range(n):
        records.append(
            {"timestamp": timestamps[i], "text": mmlu_texts[i]}
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "text"])
            writer.writeheader()
            for r in records:
                writer.writerow(r)

    return records
