from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "scripts"))

from model_utils import load_or_train_model


DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

TEST_PATH = DATA_DIR / "cleaned_test.csv"
OUTPUT_PATH = REPORTS_DIR / "submission.csv"


def main() -> None:
    model, metadata = load_or_train_model()
    test_df = pd.read_csv(TEST_PATH)

    feature_cols = metadata["feature_cols"]
    probabilities = model.predict_proba(test_df[feature_cols])[:, 1]

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "Churn": probabilities,
        }
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved submission file: {OUTPUT_PATH}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
