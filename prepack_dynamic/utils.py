from typing import List, Union
import torch
from torch import Tensor
from ortools.linear_solver import pywraplp
import binpacking
from transformers import AutoTokenizer
from model import CustomCausalLlamaModel, CustomCausalMistralModel


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


# https://developers.google.com/optimization/pack/bin_packing
def integer_program_packing(length_dict, max_bin_size):
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

    if status == pywraplp.Solver.OPTIMAL:
        result = []
        for j in data["bins"]:
            if y[j].solution_value() == 1:
                bin_dict = {}
                for i in data["items"]:
                    if x[i, j].solution_value() > 0:
                        bin_dict[i] = data["weights"][i]
                result.append(bin_dict)
    else:
        raise ("The problem does not have an optimal solution.")

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
