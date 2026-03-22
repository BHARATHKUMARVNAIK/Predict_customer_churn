from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_DATA_PATH = DATA_DIR / "cleaned_train.csv"
APP_MODEL_PATH = MODELS_DIR / "streamlit_churn_model.joblib"
APP_METADATA_PATH = MODELS_DIR / "streamlit_model_metadata.joblib"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"

TARGET_COL = "Churn"
ID_COL = "id"

RISK_BINS = [0.0, 0.30, 0.70, 1.0]
RISK_LABELS = ["Low", "Medium", "High"]


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_DATA_PATH)


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols + [ID_COL]]
    feature_cols = categorical_cols + numeric_cols

    X = X[feature_cols]
    return X, y, categorical_cols, numeric_cols


def build_model(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
            ("num", SimpleImputer(strategy="median"), numeric_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingClassifier(random_state=42, max_iter=200, learning_rate=0.08)),
        ]
    )


def build_metadata(df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> dict:
    metadata = {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "feature_cols": categorical_cols + numeric_cols,
        "category_options": {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_cols},
        "numeric_defaults": {col: float(df[col].median()) for col in numeric_cols},
        "risk_bins": RISK_BINS,
        "risk_labels": RISK_LABELS,
    }

    if FEATURE_IMPORTANCE_PATH.exists():
        metadata["feature_importance"] = pd.read_csv(FEATURE_IMPORTANCE_PATH).to_dict("records")
    else:
        metadata["feature_importance"] = []

    return metadata


def train_and_save_model() -> tuple[Pipeline, dict]:
    df = load_training_data()
    X, y, categorical_cols, numeric_cols = prepare_training_data(df)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_model(categorical_cols, numeric_cols)
    model.fit(X_train, y_train)

    metadata = build_metadata(df, categorical_cols, numeric_cols)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, APP_MODEL_PATH)
    joblib.dump(metadata, APP_METADATA_PATH)
    return model, metadata


def load_or_train_model() -> tuple[Pipeline, dict]:
    if APP_MODEL_PATH.exists() and APP_METADATA_PATH.exists():
        try:
            model = joblib.load(APP_MODEL_PATH)
            metadata = joblib.load(APP_METADATA_PATH)
            return model, metadata
        except Exception:
            pass

    return train_and_save_model()


def get_risk_level(probability: float) -> str:
    if probability <= RISK_BINS[1]:
        return "Low"
    if probability <= RISK_BINS[2]:
        return "Medium"
    return "High"


def recommend_action(customer_data: dict, probability: float) -> str:
    risk_level = get_risk_level(probability)

    if risk_level == "High":
        if customer_data.get("Contract") == "Month-to-month":
            return "Offer a contract-upgrade retention deal and contact the customer immediately."
        if customer_data.get("TechSupport") == "No" or customer_data.get("OnlineSecurity") == "No":
            return "Prioritize a retention call and offer a bundled support/security upgrade."
        if customer_data.get("MonthlyCharges", 0) >= 80:
            return "Trigger a price-relief or discount offer with a retention specialist follow-up."
        return "Escalate to the retention team with a personalized renewal or loyalty offer."

    if risk_level == "Medium":
        if customer_data.get("PaperlessBilling") == "Yes":
            return "Send a targeted retention campaign with service-value messaging and plan review."
        return "Nudge with a personalized email and highlight service benefits or add-on bundles."

    return "Keep the customer engaged with loyalty messaging and monitor for future risk changes."


def score_customer(model: Pipeline, metadata: dict, customer_data: dict) -> dict:
    payload = {}

    for col in metadata["feature_cols"]:
        if col in customer_data:
            payload[col] = customer_data[col]
        elif col in metadata["numeric_defaults"]:
            payload[col] = metadata["numeric_defaults"][col]
        else:
            payload[col] = metadata["category_options"][col][0]

    input_df = pd.DataFrame([payload])[metadata["feature_cols"]]
    probability = float(model.predict_proba(input_df)[0, 1])
    risk_level = get_risk_level(probability)
    action = recommend_action(payload, probability)

    return {
        "input_df": input_df,
        "probability": probability,
        "risk_level": risk_level,
        "action_recommendation": action,
    }
