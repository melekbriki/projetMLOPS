from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="German Credit Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class CreditData(BaseModel):
    Status: int
    Duration: float
    CreditHistory: int
    Purpose: int
    CreditAmount: float
    Savings: int
    EmploymentDuration: int
    InstallmentRate: float
    PersonalStatusSex: int
    OtherDebtors: int
    ResidenceDuration: int
    Property: int
    Age: float
    OtherInstallmentPlans: int
    Housing: int
    ExistingCredits: int
    Job: int
    PeopleLiable: int
    Telephone: int
    ForeignWorker: int

@app.get("/")
def root():
    return {"message": "German Credit Risk API is running"}

@app.post("/predict")
def predict(data: CreditData):
    input_df = pd.DataFrame([data.dict()])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[:, 1][0]
    return {
        "prediction": int(prediction[0]),
        "risk_probability": round(float(proba), 4),
        "label": "Risque élevé 🔴" if prediction[0] == 1 else "Faible risque 🟢"
    }