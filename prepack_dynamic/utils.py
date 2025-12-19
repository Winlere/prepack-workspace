from typing import List, Union, Dict, Tuple, Optional
import time

import torch
from torch import Tensor
from ortools.linear_solver import pywraplp
import binpacking
from transformers import AutoTokenizer
from model import CustomCausalLlamaModel, CustomCausalMistralModel


# 记录最近一次 ILP 求解时间（秒），便于在模拟中做 CPU/GPU overlap
LAST_ILP_SOLVE_TIME: float = 0.0


# As implemented here:
# https://github.com/pytorch/pytorch/issues/10536#issuecomment-1320935162
def left_pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = True,
    padding_value: float = 0.0,
) -> Tensor:

    sequences = tuple(map(lambda s: s.flip(0), sequences))
    padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
    _seq_dim = padded_sequence.dim()
    padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    return padded_sequence


def greedy_packing(length_dict, max_bin_size):
    return binpacking.to_constant_volume(length_dict, max_bin_size)


def _solve_prefix_bin_packing_ilp(
    prefix_items: List[Tuple[int, int]],
    max_bin_size: int,
    max_bins: int,
) -> Optional[List[Dict[int, int]]]:
    """
    Internal helper: given a prefix of items (index, length), decide if they can be
    packed into at most `max_bins` bins of capacity `max_bin_size` using ILP.

    Returns:
        - list of bins (each bin is a dict: original_index -> length) if feasible
        - None if infeasible
    """
    if max_bins <= 0:
        return None

    item_indices = [idx for idx, _ in prefix_items]
    weights = {idx: length for idx, length in prefix_items}
    bins = list(range(max_bins))

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return None

    # x[i, b] = 1 if item i is placed into bin b
    x: Dict[Tuple[int, int], pywraplp.Variable] = {}
    for i in item_indices:
        for b in bins:
            x[(i, b)] = solver.IntVar(0, 1, f"x_{i}_{b}")

    # y[b] = 1 if bin b is used (has at least one item)
    y: Dict[int, pywraplp.Variable] = {}
    for b in bins:
        y[b] = solver.IntVar(0, 1, f"y_{b}")

    # Each item must be placed in exactly one bin
    for i in item_indices:
        solver.Add(sum(x[(i, b)] for b in bins) == 1)

    # Capacity constraint for each bin
    for b in bins:
        solver.Add(
            sum(weights[i] * x[(i, b)] for i in item_indices) <= y[b] * max_bin_size
        )

    # Optional objective: minimize number of used bins (not strictly required
    # for feasibility, but helps produce compact packings).
    solver.Minimize(solver.Sum([y[b] for b in bins]))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return None

    result: List[Dict[int, int]] = []
    for b in bins:
        if y[b].solution_value() >= 0.5:
            bin_dict: Dict[int, int] = {}
            for i in item_indices:
                if x[(i, b)].solution_value() >= 0.5:
                    bin_dict[i] = weights[i]
            if bin_dict:
                result.append(bin_dict)

    return result if result else None


def prefix_ilp_packing(
    length_dict: Dict[int, int],
    max_bin_size: int,
    max_bins: Optional[int] = None,
) -> List[Dict[int, int]]:
    """
    ILP-based bin packing that respects a prefix constraint and an optional
    limit on the number of bins.

    Idea (match your description):
        - Consider items in the order of their original indices (0, 1, 2, ...).
        - We are only allowed to take a prefix {0..K-1}, cannot "skip" items.
        - Within this prefix we can arbitrarily reorder for packing.
        - We try to use at most `max_bins` bins, each with capacity `max_bin_size`.
        - We choose the *largest* prefix length K that is still feasible.

    Args:
        length_dict: mapping from original item index -> prompt length.
        max_bin_size: bin capacity C (in tokens).
        max_bins: maximum number of bins m (if None, fall back to unlimited bins).

    Returns:
        A list of bins, in the same format as `greedy_packing` /
        `integer_program_packing`: each bin is a dict mapping
        original_index -> length.

    Notes:
        - If `max_bins` is None, this reduces to solving on the full prefix
          (all items) with an effectively unlimited bin budget, but using ILP
          instead of FFD.
        - On ILP failure or if no feasible prefix is found, we fall back to
          the existing greedy packing to keep behavior robust.
    """
    if not length_dict:
        return []

    # Sort by original index so that "prefix" matches queue order.
    items_sorted: List[Tuple[int, int]] = sorted(length_dict.items(), key=lambda kv: kv[0])
    n = len(items_sorted)

    # If max_bins is not given, allow as many bins as items (effectively unlimited).
    effective_max_bins = max_bins if max_bins is not None else n

    best_packing: Optional[List[Dict[int, int]]] = None
    # We grow the prefix {0..K-1} until it becomes infeasible.
    for K in range(1, n + 1):
        prefix_items = items_sorted[:K]
        packing = _solve_prefix_bin_packing_ilp(
            prefix_items=prefix_items,
            max_bin_size=max_bin_size,
            max_bins=effective_max_bins,
        )
        if packing is None:
            # Once K is infeasible, any larger K' > K is also infeasible
            # under the same (m, C), so we can stop.
            break
        best_packing = packing

    if best_packing is not None:
        return best_packing

    # Fallback: if even K=1 is infeasible under the given (m, C)
    # or solver failed, fall back to the existing greedy packing
    # to avoid breaking the pipeline.
    return greedy_packing(length_dict, max_bin_size)


# https://developers.google.com/optimization/pack/bin_packing
def integer_program_packing(length_dict, max_bin_size):
    global LAST_ILP_SOLVE_TIME
    t0 = time.time()

    data = {}
    data["items"] = list(length_dict.keys())
    data["weights"] = list(length_dict.values())
    data["bins"] = data["items"]
    data["bin_capacity"] = max_bin_size

    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return
    x = {}
    for i in data["items"]:
        for j in data["bins"]:
            x[(i, j)] = solver.IntVar(0, 1, "x_%i_%i" % (i, j))
    y = {}
    for j in data["bins"]:
        y[j] = solver.IntVar(0, 1, "y[%i]" % j)

    for i in data["items"]:
        solver.Add(sum(x[i, j] for j in data["bins"]) == 1)

    for j in data["bins"]:
        solver.Add(sum(x[(i, j)] * data["weights"][i] for i in data["items"]) <= y[j] * data["bin_capacity"])

    solver.Minimize(solver.Sum([y[j] for j in data["bins"]]))

    status = solver.Solve()
    solve_time = time.time() - t0
    LAST_ILP_SOLVE_TIME = solve_time

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        result = []
        for j in data["bins"]:
            if y[j].solution_value() == 1:
                bin_dict = {}
                for i in data["items"]:
                    if x[i, j].solution_value() > 0:
                        bin_dict[i] = data["weights"][i]
                result.append(bin_dict)
        # 打印 ILP 求解时间以及本次打包的规模，方便和 TTFT 对比
        print(
            f"[ILP-PACK] n_items={len(data['items'])}, "
            f"max_bin_size={max_bin_size}, "
            f"solve_time={solve_time:.6f}s"
        )    
    elif status == pywraplp.Solver.INFEASIBLE:
        raise Exception("The problem is infeasible.")
    else:
        raise Exception("The problem does not have a valid solution.")

    return result


class AIMDBatchController:
    """
    Simple AIMD (Additive Increase / Multiplicative Decrease) controller for batch size.

    The controller maintains an internal batch size and updates it based on the
    observed latency of the previous batch.
    """

    def __init__(
        self,
        init_bs: int,
        min_bs: int,
        max_bs: int,
        target_latency: float,
        alpha: float = 1.0,
        beta: float = 0.5,
    ):
        self.init_bs = int(init_bs)
        self.min_bs = int(min_bs)
        self.max_bs = int(max_bs)
        self.target_latency = float(target_latency)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.batch_size = max(self.min_bs, min(self.init_bs, self.max_bs))

    def update(self, last_latency: float) -> int:
        """
        Update the internal batch size using AIMD based on the last latency.

        If last_latency < target_latency: increase additively by alpha.
        Otherwise: decrease multiplicatively by beta.
        The resulting batch size is clamped to [min_bs, max_bs].
        """
        if last_latency < self.target_latency:
            self.batch_size += self.alpha
        else:
            # Multiplicative decrease
            self.batch_size = self.batch_size * self.beta

        # Ensure integer batch size
        self.batch_size = int(max(self.min_bs, min(self.batch_size, self.max_bs)))
        if self.batch_size < 1:
            self.batch_size = 1
        return self.batch_size


def load_model_and_tokenizer(
    base_model: str = "llama1b",
    loadbit: int = 8,
):
    # Load tokenizer and model
    if base_model == "llama1b":
        path = "princeton-nlp/Sheared-LLaMA-1.3B"
    elif base_model == "llama2.7b":
        path = "princeton-nlp/Sheared-LLaMA-2.7B"

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = "[PAD]"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_in_8bit = loadbit == 8
    load_in_4bit = loadbit == 4
    if "llama" in base_model:
        model = CustomCausalLlamaModel.from_pretrained(
            path, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
    elif "mistral" in base_model:
        model = CustomCausalMistralModel.from_pretrained(
            path, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
    model.eval()
    if loadbit != 8 and loadbit != 4:
        model.to(device)

    return model, tokenizer
