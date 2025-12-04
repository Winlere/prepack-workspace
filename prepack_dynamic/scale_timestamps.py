import csv
import os

import fire


def main(
    input_path: str = "/u/wzhan/prepack-workspace/dataset/combined/mmlu_azure_ts.csv",
    output_path: str = "/u/wzhan/prepack-workspace/dataset/combined/mmlu_azure_ts_scaled_4.csv",
    scale: float = 4.0,
):
    """
    Scale timestamps in a CSV by dividing them by `scale`.

    The CSV is expected to have at least two columns:
      - 'timestamp'
      - 'text'

    Output CSV will have the same columns, but with:
      new_timestamp = old_timestamp / scale
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r") as fin, open(output_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if fieldnames is None or "timestamp" not in fieldnames:
            raise ValueError("Input CSV must have a 'timestamp' column.")

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                t = float(row["timestamp"])
            except (TypeError, ValueError):
                continue
            row["timestamp"] = t / scale
            writer.writerow(row)

    print(f"Scaled timestamps by 1/{scale} and saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)


