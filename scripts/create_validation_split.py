import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from sklearn.model_selection import train_test_split

from RM_LogisticRegression.data.processing import label_names, print_label_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible stratified train/validation split."
    )
    parser.add_argument("--input", default="data/train.csv", help="Source train CSV.")
    parser.add_argument(
        "--train-output",
        default="data/train_split.csv",
        help="Path for the training split CSV.",
    )
    parser.add_argument(
        "--valid-output",
        default="data/valid_split.csv",
        help="Path for the validation split CSV.",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.1,
        help="Validation fraction. Use 0.1 for a 90/10 split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    train_output = Path(args.train_output)
    valid_output = Path(args.valid_output)

    df = pd.read_csv(input_path)
    labels = label_names(df)

    train_df, valid_df = train_test_split(
        df,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=labels,
        shuffle=True,
    )

    train_output.parent.mkdir(parents=True, exist_ok=True)
    valid_output.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_output, index=False)
    valid_df.to_csv(valid_output, index=False)

    print(f"Wrote {train_output}")
    print(f"Wrote {valid_output}")
    print_label_distribution("full", df)
    print_label_distribution("train", train_df)
    print_label_distribution("valid", valid_df)


if __name__ == "__main__":
    main()

