from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

#  FastAPI instance
app = FastAPI(title="Churn Prediction API")

# Correct model path
models_path = Path('./models/churn_pipeline.pkl').resolve().parents[1] / "models" / "churn_pipeline.pkl"
churn_pipeline = joblib.load(models_path)

print(models_path)
print(models_path.exists())

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


# Input data model
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    df = pd.DataFrame([features.model_dump()])

    prob = churn_pipeline.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)

    return {
        "churn_probability": float(prob),
        "churn_predicted": pred
    }
