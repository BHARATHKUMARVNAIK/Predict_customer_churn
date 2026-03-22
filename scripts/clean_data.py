from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
OUTPUT_TRAIN_PATH = DATA_DIR / "cleaned_train.csv"
OUTPUT_TEST_PATH = DATA_DIR / "cleaned_test.csv"

TARGET_COL = "Churn"
ID_COL = "id"
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
OBJECT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
INTERNET_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in OBJECT_COLS + ([TARGET_COL] if TARGET_COL in cleaned.columns else []):
        cleaned[col] = cleaned[col].astype(str).str.strip()
    return cleaned


def validate_dataset(df: pd.DataFrame, name: str) -> None:
    missing_counts = df.isna().sum()
    if missing_counts.any():
        raise ValueError(f"{name} still has missing values:\n{missing_counts[missing_counts > 0]}")

    duplicate_count = df.duplicated().sum()
    if duplicate_count:
        raise ValueError(f"{name} still has {duplicate_count} duplicate rows.")

    if not df[ID_COL].is_unique:
        raise ValueError(f"{name} has duplicate ids.")

    invalid_phone_rows = (df["PhoneService"].eq("No") & df["MultipleLines"].ne("No phone service")).sum()
    if invalid_phone_rows:
        raise ValueError(f"{name} has {invalid_phone_rows} inconsistent phone-service rows.")

    invalid_internet_rows = (
        df["InternetService"].eq("No")
        & ~df[INTERNET_SERVICE_COLS].eq("No internet service").all(axis=1)
    ).sum()
    if invalid_internet_rows:
        raise ValueError(f"{name} has {invalid_internet_rows} inconsistent internet-service rows.")

    if (df["tenure"] < 0).any() or (df["MonthlyCharges"] < 0).any() or (df["TotalCharges"] < 0).any():
        raise ValueError(f"{name} contains invalid negative numeric values.")


def clean_dataset(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    cleaned = normalize_text_columns(df)

    for col in NUMERIC_COLS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="raise")

    cleaned[ID_COL] = pd.to_numeric(cleaned[ID_COL], errors="raise").astype("int64")
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype("int8")
    cleaned["tenure"] = cleaned["tenure"].astype("int16")

    if is_train:
        churn_map = {"No": 0, "Yes": 1}
        unexpected = sorted(set(cleaned[TARGET_COL]) - set(churn_map))
        if unexpected:
            raise ValueError(f"Unexpected churn values: {unexpected}")
        cleaned[TARGET_COL] = cleaned[TARGET_COL].map(churn_map).astype("int8")

    validate_dataset(cleaned, "train" if is_train else "test")
    return cleaned


def main() -> None:
    train_df = clean_dataset(load_dataset(TRAIN_PATH), is_train=True)
    test_df = clean_dataset(load_dataset(TEST_PATH), is_train=False)

    train_df.to_csv(OUTPUT_TRAIN_PATH, index=False)
    test_df.to_csv(OUTPUT_TEST_PATH, index=False)

    print(f"Saved {OUTPUT_TRAIN_PATH.name}: {train_df.shape}")
    print(f"Saved {OUTPUT_TEST_PATH.name}: {test_df.shape}")


if __name__ == "__main__":
    main()
