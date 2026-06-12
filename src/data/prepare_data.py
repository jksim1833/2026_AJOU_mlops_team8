from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


RAW_PATH = Path("data/raw/DieCasting_product1.csv")
PROCESSED_DIR = Path("data/processed")

ORIGINAL_LABEL_COLUMN = "label"
TARGET_COLUMN = "defect_label"
RANDOM_STATE = 42


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.strip()

    print("===== Raw data info =====")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    if ORIGINAL_LABEL_COLUMN not in df.columns:
        raise ValueError(f"'{ORIGINAL_LABEL_COLUMN}' column not found.")

    print("\n===== Original label distribution =====")
    print(df[ORIGINAL_LABEL_COLUMN].value_counts().sort_index())

    print("\n===== Missing values by column =====")
    print(df.isnull().sum())

    # label 기준:
    # 0 = normal
    # 1, 2 = defect
    df[TARGET_COLUMN] = (df[ORIGINAL_LABEL_COLUMN] != 0).astype(int)

    print("\n===== Binary label distribution =====")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    df = df.drop(columns=[ORIGINAL_LABEL_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("\n===== Before oversampling =====")
    print("train:", y_train.value_counts().sort_index().to_dict())
    print("valid:", y_valid.value_counts().sort_index().to_dict())
    print("test:", y_test.value_counts().sort_index().to_dict())

    oversampler = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_os, y_train_os = oversampler.fit_resample(X_train, y_train)

    train = X_train_os.copy()
    train[TARGET_COLUMN] = y_train_os

    valid = X_valid.copy()
    valid[TARGET_COLUMN] = y_valid

    test = X_test.copy()
    test[TARGET_COLUMN] = y_test

    binary = df.copy()

    binary.to_csv(PROCESSED_DIR / "diecasting_product1_binary.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    valid.to_csv(PROCESSED_DIR / "valid.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("\n===== After oversampling =====")
    print("train:", train[TARGET_COLUMN].value_counts().sort_index().to_dict())
    print("valid:", valid[TARGET_COLUMN].value_counts().sort_index().to_dict())
    print("test:", test[TARGET_COLUMN].value_counts().sort_index().to_dict())

    print("\n===== Saved files =====")
    print(PROCESSED_DIR / "diecasting_product1_binary.csv")
    print(PROCESSED_DIR / "train.csv")
    print(PROCESSED_DIR / "valid.csv")
    print(PROCESSED_DIR / "test.csv")


if __name__ == "__main__":
    main()
