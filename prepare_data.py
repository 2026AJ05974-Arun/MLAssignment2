"""Download the UCI Letter Recognition data and split it into train/test CSVs.

Run it with:  python prepare_data.py
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
TARGET = "Class"
FEATURES = [
    "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar", "x2bar",
    "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx",
]


def load_data():
    # the zip has a headerless file, first column is the letter
    with urllib.request.urlopen(URL, timeout=60) as response:
        zip_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with z.open("letter-recognition.data") as f:
            return pd.read_csv(f, header=None, names=[TARGET] + FEATURES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=20261106)
    args = parser.parse_args()

    data = load_data()

    # quick sanity checks before splitting
    assert len(data) >= 500, "not enough rows"
    assert data[TARGET].nunique() == 26, "expected 26 letters"
    assert not data.isna().any().any(), "found missing values"

    train, test = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data[TARGET],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train_data.csv", index=False)
    test.to_csv(args.output_dir / "test_data.csv", index=False)

    print(f"{len(data)} rows, {len(FEATURES)} features, {data[TARGET].nunique()} classes")
    print(f"train: {len(train)} rows, test: {len(test)} rows")


if __name__ == "__main__":
    main()
