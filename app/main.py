import io
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Basic setup
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
STATIC_DIR = ROOT_DIR / "static"

REQUIRED_COLUMNS = [
    "age",
    "gender",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "device",
    "monthly_fee",
    "payment_method",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "favorite_genre",
]


app = FastAPI(title="Loyalty Lens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# 2. Input format for manual prediction
# ---------------------------------------------------------------------------

class CustomerInput(BaseModel):
    age: int = Field(..., ge=0, le=100)
    gender: Literal["Male", "Female", "Other"]
    subscription_type: Literal["Basic", "Standard", "Premium"]
    watch_hours: float = Field(..., ge=0)
    last_login_days: int = Field(..., ge=0)
    device: Literal["Mobile", "TV", "Laptop", "Tablet", "Desktop"]
    monthly_fee: float = Field(..., ge=0)
    payment_method: Literal["Gift Card", "Crypto", "Debit Card", "PayPal", "Credit Card"]
    number_of_profiles: int = Field(..., ge=0, le=5)
    avg_watch_time_per_day: float = Field(..., ge=0)
    favorite_genre: Literal["Action", "Sci-Fi", "Drama", "Horror", "Romance", "Comedy", "Documentary"]


# ---------------------------------------------------------------------------
# 3. Load trained model and preprocessing files once when the API starts
# ---------------------------------------------------------------------------

model = joblib.load(ARTIFACT_DIR / "voting_classifier_model.joblib")
encoders = joblib.load(ARTIFACT_DIR / "encoders.joblib")
imputers = joblib.load(ARTIFACT_DIR / "imputers.joblib")
feature_columns = joblib.load(ARTIFACT_DIR / "feature_columns.joblib")
background = pd.read_csv(ARTIFACT_DIR / "shap_background.csv")

# Remove the target column if it is present in saved preprocessing objects.
feature_columns = [col for col in feature_columns if col != "churned"]
encoders.pop("churned", None)
imputers.pop("churned", None)

# The Streamlit app used the Gradient Boosting estimator for SHAP explanations.
gtb_model = model.named_estimators_["gtb"]
explainer = shap.Explainer(gtb_model, background, model_output="probability")


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same imputation, encoding, and column order used in Streamlit."""
    df = df.copy()

    for col, imputer in imputers.items():
        if col in df.columns:
            df[col] = imputer.transform(df[[col]]).ravel()

    for col, encoder in encoders.items():
        if col in df.columns:
            encoded = encoder.transform(df[[col]])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out([col]),
                index=df.index,
            )
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

    return df.reindex(columns=feature_columns, fill_value=0)


def readable_name(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def shap_table(x_data: pd.DataFrame) -> pd.DataFrame:
    values = explainer(x_data).values
    if values.ndim == 3:
        values = values[:, :, 1]

    df = pd.DataFrame({"feature": x_data.columns, "shap_value": values[0]})
    df["absolute_magnitude"] = df["shap_value"].abs()
    df["feature"] = df["feature"].apply(readable_name)

    churn_index = list(model.classes_).index(0)
    df["impact"] = df["shap_value"].apply(
        lambda value: "Which Increases churn risk"
        if (value < 0 if churn_index == 0 else value > 0)
        else "Which Reduces churn risk"
    )

    return df.sort_values("absolute_magnitude", ascending=False)


def drivers_from_shap(shap_df: pd.DataFrame, status: str) -> list[dict]:
    if status == "HIGH CHURN RISK":
        drivers = shap_df[shap_df["impact"].str.contains("Increases")].head(3)
    else:
        drivers = shap_df[shap_df["impact"].str.contains("Reduces")].head(3)

    if drivers.empty:
        drivers = shap_df.head(3)

    return [
        {
            "feature": row["feature"],
            "impact": row["impact"],
            "absolute_magnitude": float(row["absolute_magnitude"]),
            "raw_weight_value": float(row["shap_value"]),
        }
        for _, row in drivers.iterrows()
    ]


def row_explanation(feature_names: list[str], row_values: np.ndarray, status: str, confidence: float) -> str:
    top_indexes = np.argsort(np.abs(row_values))[-3:][::-1]
    top_features = [readable_name(feature_names[index]) for index in top_indexes]
    return f"{status} with {confidence:.2%} confidence. Main factors: {', '.join(top_features)}."


def predict_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x_data = preprocess(df)
    predictions = model.predict(x_data).astype(int)
    probabilities = model.predict_proba(x_data)
    return x_data, predictions, probabilities


# ---------------------------------------------------------------------------
# 5. API routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "required_columns": REQUIRED_COLUMNS,
        "classes": [int(label) for label in model.classes_],
    }


@app.post("/predict")
def predict_single(customer: CustomerInput):
    input_df = pd.DataFrame([customer.model_dump()])
    x_data, predictions, probabilities = predict_dataframe(input_df)

    prediction = int(predictions[0])
    churn_index = list(model.classes_).index(0)
    retention_index = list(model.classes_).index(1)

    churn_probability = float(probabilities[0][churn_index])
    retention_probability = float(probabilities[0][retention_index])

    if prediction == 0:
        status = "HIGH CHURN RISK"
        confidence = churn_probability
    else:
        status = "LOW CHURN RISK"
        confidence = retention_probability

    shap_df = shap_table(x_data)
    top_drivers = drivers_from_shap(shap_df, status)
    reason_text = "; ".join(driver["feature"] for driver in top_drivers)

    return {
        "prediction": prediction,
        "status": status,
        "confidence": confidence,
        "churn_probability": churn_probability,
        "retention_probability": retention_probability,
        "top_drivers": top_drivers,
        "explanation": f"{status} with {confidence:.2%} confidence. Main factors: {reason_text}.",
    }


@app.post("/predict-bulk")
async def predict_bulk(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    content = await file.read()

    try:
        if filename.endswith(".csv"):
            input_df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            input_df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Please upload a CSV or Excel file.")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read file: {error}") from error

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in input_df.columns]
    if missing_columns:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing_columns}")

    x_data, predictions, probabilities = predict_dataframe(input_df)
    churn_index = list(model.classes_).index(0)
    retention_index = list(model.classes_).index(1)

    shap_values = explainer(x_data).values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    records = []
    for index, prediction in enumerate(predictions):
        churn_probability = float(probabilities[index][churn_index])
        retention_probability = float(probabilities[index][retention_index])
        status = "HIGH CHURN RISK" if prediction == 0 else "LOW CHURN RISK"
        confidence = churn_probability if prediction == 0 else retention_probability
        row_data = input_df.iloc[index][REQUIRED_COLUMNS].replace({np.nan: None}).to_dict()

        records.append(
            {
                "row_number": index + 1,
                **row_data,
                "prediction": int(prediction),
                "status": status,
                "churn_probability": churn_probability,
                "confidence": confidence,
                "explanation": row_explanation(list(x_data.columns), shap_values[index], status, confidence),
            }
        )

    global_drivers = pd.DataFrame(
        {
            "feature": [readable_name(col) for col in x_data.columns],
            "importance": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("importance", ascending=False).head(3)

    churn_count = int((predictions == 0).sum())

    return {
        "total_records": len(input_df),
        "likely_churned_profiles": churn_count,
        "churn_rate": churn_count / len(input_df),
        "global_top_drivers": global_drivers.to_dict(orient="records"),
        "records": records,
    }
