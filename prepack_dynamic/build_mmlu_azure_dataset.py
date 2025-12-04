import fire

from utils import load_model_and_tokenizer
from dataset_utils import build_mmlu_azure_dataset


def main(
    trace_csv_path: str = "/u/wzhan/prepack-workspace/dataset/microsoft/AzureLLMInferenceTrace_conv_1week.csv",
    output_path: str = "/u/wzhan/prepack-workspace/dataset/combined/mmlu_azure_ts.csv",
    sample_size: int = 1000,
    model_name: str = "llama1b",
    loadbit: int = 4,
):
    """
    Build and persist a combined dataset: MMLU questions + Azure timestamps.

    This script:
      1) Loads the tokenizer (no need to keep the model in memory afterwards).
      2) Fetches MMLU questions (sample_size).
      3) Reads Azure trace timestamps (and token stats) from `trace_csv_path`.
      4) Aligns the first N entries and writes a CSV with columns:
         [timestamp, text, context_tokens, generated_tokens]
         to `output_path`.

    After running once, you can reuse the saved CSV without rebuilding it each time.
    """

    # Only need tokenizer; model can be discarded immediately after load.
    model, tokenizer = load_model_and_tokenizer(base_model=model_name, loadbit=loadbit)
    del model

    records = build_mmlu_azure_dataset(
        tokenizer=tokenizer,
        trace_csv_path=trace_csv_path,
        sample_size=sample_size,
        save_path=output_path,
    )

    print(f"Saved combined dataset with {len(records)} records to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)


