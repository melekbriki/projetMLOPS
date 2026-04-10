
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="German Credit Risk API")

@app.get("/")
def root():
    return {"message": "German Credit Risk API is running ✅"}
