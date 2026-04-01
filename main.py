"""
main.py — German Credit Risk API
FastAPI backend complet pour le projet MLOps
"""

import os
import uuid
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# ── Constantes ────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "EmploymentDuration", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "ResidenceDuration", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "ExistingCredits",
    "Job", "PeopleLiable", "Telephone", "ForeignWorker",
]

DATA_FILE   = Path("german.data")
MODELS_DIR  = Path("saved_models")
HISTORY_FILE = Path("history.json")
UPLOAD_DIR  = Path("uploads")

MODELS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ── App & CORS ────────────────────────────────────────────────────────────────

app = FastAPI(title="German Credit Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── État global ───────────────────────────────────────────────────────────────

_trained_models:  Dict[str, Any]            = {}
_trained_scalers: Dict[str, StandardScaler] = {}
_train_data:      Optional[tuple]           = None   # (X_train_bal, y_train_bal, X_test, y_test)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_history() -> List[dict]:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def _save_history(runs: List[dict]) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(runs, f, indent=2)

def _prepare_data(df: pd.DataFrame):
    """Pipeline complet : encode → scale → split → SMOTE."""
    df = df.copy()
    df["Target"] = df["Target"].map({1: 0, 2: 1})

    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Target", axis=1)
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    return X_train_bal, y_train_bal, X_test, y_test, scaler

def _build_model(model_id: str, hyperparams: dict):
    """Instanciation du modèle sklearn selon l'identifiant."""
    hp = hyperparams or {}

    if model_id == "logistic_regression":
        return LogisticRegression(
            C=float(hp.get("C", 1.0)),
            max_iter=int(hp.get("max_iter", 200)),
            solver=hp.get("solver", "lbfgs"),
            random_state=42,
        )
    elif model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 100)),
            max_depth=int(hp.get("max_depth", 10)) if hp.get("max_depth") else None,
            min_samples_split=int(hp.get("min_samples_split", 2)),
            random_state=42,
        )
    elif model_id == "svm":
        return SVC(
            C=float(hp.get("C", 1.0)),
            kernel=hp.get("kernel", "rbf"),
            gamma=hp.get("gamma", "scale"),
            probability=True,
            random_state=42,
        )
    elif model_id == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(hp.get("n_neighbors", 5)),
            weights=hp.get("weights", "uniform"),
            metric=hp.get("metric", "minkowski"),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Modèle inconnu : {model_id}")

def _compute_metrics(model, X_test, y_test):
    """Calcule toutes les métriques + confusion matrix + ROC."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred) * 100, 2),
        "roc_auc":   round(roc_auc_score(y_test, y_proba) * 100, 2),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": [round(x, 4) for x in fpr.tolist()],
            "tpr": [round(x, 4) for x in tpr.tolist()],
        },
    }

def _ensure_data():
    """Charge et prépare les données une seule fois."""
    global _train_data
    if _train_data is not None:
        return _train_data

    # 1. Chercher dans les uploads en priorité
    uploaded = list(UPLOAD_DIR.glob("*.csv")) + list(UPLOAD_DIR.glob("*.data"))
    if uploaded:
        src = uploaded[-1]
        df = pd.read_csv(src, sep=r"\s+", header=None) if src.suffix == ".data" \
             else pd.read_csv(src)
    elif DATA_FILE.exists():
        cols = FEATURE_COLUMNS + ["Target"]
        df = pd.read_csv(DATA_FILE, sep=" ", header=None, names=cols)
    else:
        raise HTTPException(
            status_code=503,
            detail="Aucun dataset trouvé. Uploadez un fichier CSV via /upload."
        )

    if "Target" not in df.columns:
        raise HTTPException(status_code=422, detail="Colonne 'Target' introuvable dans le dataset.")

    _train_data = _prepare_data(df)
    return _train_data

# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model: str
    hyperparams: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    model_id: str
    features: Dict[str, float]

class TuneRequest(BaseModel):
    model: str
    method: str = "GridSearch"   # GridSearch | RandomSearch

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "German Credit Risk API ✅"}


@app.get("/models")
def list_models():
    """Liste les modèles disponibles et ceux déjà entraînés."""
    available = [
        {"id": "logistic_regression", "name": "Logistic Regression", "trained": "logistic_regression" in _trained_models},
        {"id": "random_forest",       "name": "Random Forest",       "trained": "random_forest"       in _trained_models},
        {"id": "svm",                 "name": "SVM",                 "trained": "svm"                 in _trained_models},
        {"id": "knn",                 "name": "KNN",                 "trained": "knn"                 in _trained_models},
    ]
    return {"models": available}


@app.post("/train")
def train(req: TrainRequest):
    """Entraîne un modèle avec les hyperparamètres fournis."""
    X_train_bal, y_train_bal, X_test, y_test, scaler = _ensure_data()

    model = _build_model(req.model, req.hyperparams)
    model.fit(X_train_bal, y_train_bal)

    metrics = _compute_metrics(model, X_test, y_test)

    # Sauvegarde en mémoire + disque
    _trained_models[req.model]  = model
    _trained_scalers[req.model] = scaler

    model_path = MODELS_DIR / f"{req.model}_model.pkl"
    joblib.dump(model,  model_path)
    joblib.dump(scaler, MODELS_DIR / f"{req.model}_scaler.pkl")

    # Historique
    run = {
        "id":      str(uuid.uuid4()),
        "model":   req.model,
        "date":    datetime.now().strftime("%d/%m/%Y %H:%M"),
        "params":  req.hyperparams,
        **metrics,
    }
    runs = _load_history()
    runs.insert(0, run)
    _save_history(runs)

    return {"model": req.model, **metrics, "run_id": run["id"]}


@app.get("/results")
def get_results(model_id: str = Query(...)):
    """Retourne les métriques d'un modèle déjà entraîné."""
    if model_id not in _trained_models:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' non entraîné.")
    _, _, X_test, y_test, _ = _ensure_data()
    metrics = _compute_metrics(_trained_models[model_id], X_test, y_test)
    return {"model": model_id, **metrics}


@app.post("/predict")
def predict(req: PredictRequest):
    """Effectue une prédiction pour un demandeur de crédit."""
    model_id = req.model_id

    # Charger depuis disque si pas en mémoire
    if model_id not in _trained_models:
        model_path  = MODELS_DIR / f"{model_id}_model.pkl"
        scaler_path = MODELS_DIR / f"{model_id}_scaler.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' non trouvé. Entraînez-le d'abord.")
        _trained_models[model_id]  = joblib.load(model_path)
        _trained_scalers[model_id] = joblib.load(scaler_path)

    model  = _trained_models[model_id]
    scaler = _trained_scalers[model_id]

    # Construire le DataFrame dans le bon ordre de colonnes
    try:
        input_df = pd.DataFrame([{col: req.features.get(col, 0) for col in FEATURE_COLUMNS}])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Features invalides : {e}")

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)
    proba        = model.predict_proba(input_scaled)[:, 1][0]

    return {
        "prediction":       int(prediction[0]),
        "risk_probability": round(float(proba), 4),
        "model_used":       model_id,
    }


@app.post("/tune")
def auto_tune(req: TuneRequest):
    """Recherche automatique des meilleurs hyperparamètres."""
    X_train_bal, y_train_bal, X_test, y_test, scaler = _ensure_data()

    param_grids = {
        "logistic_regression": {"C": [0.01, 0.1, 1, 10], "max_iter": [100, 200]},
        "random_forest":       {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
        "svm":                 {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "knn":                 {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    }

    if req.model not in param_grids:
        raise HTTPException(status_code=400, detail=f"Modèle '{req.model}' non supporté pour le tuning.")

    base_model = _build_model(req.model, {})
    grid       = param_grids[req.model]

    if req.method == "RandomSearch":
        search = RandomizedSearchCV(base_model, grid, n_iter=8, cv=3, scoring="roc_auc", random_state=42, n_jobs=-1)
    else:
        search = GridSearchCV(base_model, grid, cv=3, scoring="roc_auc", n_jobs=-1)

    search.fit(X_train_bal, y_train_bal)
    best_model = search.best_estimator_

    metrics = _compute_metrics(best_model, X_test, y_test)

    _trained_models[req.model]  = best_model
    _trained_scalers[req.model] = scaler
    joblib.dump(best_model, MODELS_DIR / f"{req.model}_model.pkl")
    joblib.dump(scaler,     MODELS_DIR / f"{req.model}_scaler.pkl")

    return {
        "model":        req.model,
        "best_params":  search.best_params_,
        "best_cv_score": round(search.best_score_ * 100, 2),
        **metrics,
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload un fichier CSV pour remplacer le dataset par défaut."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .csv sont acceptés.")

    dest = UPLOAD_DIR / "dataset.csv"
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # Réinitialiser les données pour forcer le rechargement
    global _train_data
    _train_data = None

    return {"message": f"Dataset '{file.filename}' uploadé avec succès ✅", "path": str(dest)}


@app.get("/history")
def get_history():
    """Retourne l'historique de tous les entraînements."""
    return _load_history()


@app.get("/download")
def download_model(model_id: str = Query(...)):
    """Télécharge le fichier .pkl d'un modèle entraîné."""
    path = MODELS_DIR / f"{model_id}_model.pkl"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Modèle '{model_id}' non trouvé sur disque.")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{model_id}_model.pkl")


# ── Lancement ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)