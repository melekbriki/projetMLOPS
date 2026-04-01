"""
train.py — Script d'entraînement autonome avec MLflow
Tâche 3 : Expérimentation et comparaison des algorithmes ML

Usage :
    python train.py                        # entraîne tous les modèles
    python train.py --model random_forest  # un seul modèle
    python train.py --tune                 # avec GridSearchCV
"""

import argparse
import json
import joblib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.decomposition import PCA

# ── Configuration MLflow ──────────────────────────────────────────────────────

mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "German_Credit_Risk"
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Chemins ───────────────────────────────────────────────────────────────────

DATA_PATH  = Path("german.data")
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "EmploymentDuration", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "ResidenceDuration", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "ExistingCredits",
    "Job", "PeopleLiable", "Telephone", "ForeignWorker", "Target",
]

# ── Modèles et hyperparamètres à tester ──────────────────────────────────────

MODELS_CONFIG = {
    "logistic_regression": {
        "model":  LogisticRegression(max_iter=200, random_state=42),
        "params": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    },
    "random_forest": {
        "model":  RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    },
    "svm": {
        "model":  SVC(probability=True, random_state=42),
        "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    },
    "knn": {
        "model":  KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    },
}

# ── Chargement et préparation ─────────────────────────────────────────────────

def load_and_prepare():
    print("📂 Chargement des données...")
    df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=FEATURE_COLUMNS)

    # Fix target : 1→0 (bon payeur), 2→1 (mauvais payeur)
    df["Target"] = df["Target"].map({1: 0, 2: 1})

    # Encodage des variables catégorielles
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Target", axis=1)
    y = df["Target"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE pour rééquilibrage
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"✅ Train : {X_train_bal.shape} | Test : {X_test.shape}")
    print(f"   Distribution train après SMOTE : {dict(pd.Series(y_train_bal).value_counts())}")

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    return X_train_bal, y_train_bal, X_test, y_test, scaler, X_scaled, y


def compute_metrics(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "f1":        round(f1_score(y_test, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
    }

# ── Entraînement simple ───────────────────────────────────────────────────────

def train_model(name, config, X_train, y_train, X_test, y_test):
    print(f"\n🤖 Entraînement : {name}")
    model = config["model"]

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("training_type", "default")

        # Log des hyperparamètres par défaut
        mlflow.log_params(model.get_params())

        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)

        # Log métriques MLflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Enregistrer le modèle
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"credit_{name}",
        )

        run_id = mlflow.active_run().info.run_id

    joblib.dump(model, MODELS_DIR / f"{name}_model.pkl")

    print(f"   Accuracy : {metrics['accuracy']*100:.1f}%  |  "
          f"F1 : {metrics['f1']*100:.1f}%  |  "
          f"AUC : {metrics['roc_auc']*100:.1f}%")
    print(f"   MLflow run_id : {run_id}")

    return metrics, run_id


# ── Entraînement avec GridSearch ──────────────────────────────────────────────

def train_with_gridsearch(name, config, X_train, y_train, X_test, y_test):
    print(f"\n🔍 GridSearchCV : {name}")

    with mlflow.start_run(run_name=f"{name}_GridSearch"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("training_type", "GridSearch")

        gs = GridSearchCV(
            config["model"], config["params"],
            cv=5, scoring="roc_auc", n_jobs=-1, verbose=0,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("best_cv_roc_auc", gs.best_score_)

        metrics = compute_metrics(best, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            registered_model_name=f"credit_{name}_tuned",
        )
        run_id = mlflow.active_run().info.run_id

    joblib.dump(best, MODELS_DIR / f"{name}_tuned_model.pkl")

    print(f"   Meilleurs params : {gs.best_params_}")
    print(f"   CV AUC  : {gs.best_score_*100:.1f}%  |  "
          f"Test AUC : {metrics['roc_auc']*100:.1f}%  |  "
          f"F1 : {metrics['f1']*100:.1f}%")
    print(f"   MLflow run_id : {run_id}")

    return metrics, run_id, gs.best_params_


# ── PCA pour visualisation ────────────────────────────────────────────────────

def run_pca_analysis(X_scaled, y):
    print("\n📊 Analyse PCA...")
    with mlflow.start_run(run_name="PCA_analysis"):
        mlflow.set_tag("analysis_type", "dimensionality_reduction")

        pca = PCA(n_components=20, random_state=42)
        pca.fit(X_scaled)

        cumvar = pca.explained_variance_ratio_.cumsum()
        n_95   = int(np.argmax(cumvar >= 0.95)) + 1
        n_90   = int(np.argmax(cumvar >= 0.90)) + 1

        mlflow.log_metric("components_for_95pct_variance", n_95)
        mlflow.log_metric("components_for_90pct_variance", n_90)
        mlflow.log_metric("variance_explained_2d",
                           float(pca.explained_variance_ratio_[:2].sum()))

        joblib.dump(pca, MODELS_DIR / "pca.pkl")
        print(f"   Composantes pour 95% variance : {n_95}")
        print(f"   Variance expliquée en 2D      : {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")

# ── Tableau comparatif ────────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    print("\n" + "="*75)
    print("📊 TABLEAU COMPARATIF DES MODÈLES")
    print("="*75)
    header = f"{'Modèle':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}"
    print(header)
    print("-"*75)
    for name, m in results.items():
        print(f"{name:<30} {m['accuracy']*100:>8.1f}% {m['precision']*100:>9.1f}% "
              f"{m['recall']*100:>7.1f}% {m['f1']*100:>7.1f}% {m['roc_auc']*100:>7.1f}%")
    print("="*75)

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n🏆 Meilleur modèle (AUC) : {best} — AUC = {results[best]['roc_auc']*100:.1f}%")

    # Sauvegarde JSON
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Résultats sauvegardés dans comparison_results.json")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train credit risk models")
    parser.add_argument("--model",  default="all", help="Modèle à entraîner (all par défaut)")
    parser.add_argument("--tune",   action="store_true", help="Activer GridSearchCV")
    parser.add_argument("--pca",    action="store_true", help="Analyse PCA")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, scaler, X_scaled, y = load_and_prepare()

    if args.pca:
        run_pca_analysis(X_scaled, y)

    models_to_run = MODELS_CONFIG if args.model == "all" else {
        args.model: MODELS_CONFIG[args.model]
    }

    all_results = {}

    for name, config in models_to_run.items():
        if args.tune:
            metrics, run_id, best_params = train_with_gridsearch(
                name, config, X_train, y_train, X_test, y_test
            )
        else:
            metrics, run_id = train_model(
                name, config, X_train, y_train, X_test, y_test
            )
        all_results[name] = metrics

    if len(all_results) > 1:
        print_comparison_table(all_results)

    print(f"\n✅ Terminé ! Visualisez les runs : mlflow ui --port 5000")


if __name__ == "__main__":
    main()