"""
entrenamiento.py -- Entrenamiento del modelo con XGBoost, Optuna y MLflow
"""

import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Configuración de MLflow local
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "XGBoost_Optuna_Training"
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data():
    print("\n" + "="*50)
    print("¡SÍ! ESTE ES EL ARCHIVO NUEVO Y CORRECTO")
    print("="*50 + "\n")
    
    print("[*] Cargando datos procesados...")
    df_train = pd.read_csv("data/processed/df_train.csv")
    df_test = pd.read_csv("data/processed/df_test.csv")
    
    cols_to_drop = ["target", "p_fecinformacion", "key_value", "monto", "grp_campecs06m", "prob_value_contact"]
    
    X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    y_train = df_train["target"]
    
    X_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
    y_test = df_test["target"]
    
    
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)
    
    return X_train, y_train, X_test, y_test

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "eval_metric": "auc"
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    preds_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds_proba)

def train_and_log(train_path, test_path, val_path):
    X_train, y_train, X_test, y_test = load_data()
    
    print("[*] Iniciando optimización con Optuna (30 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)
    
    print(f"\n[*] Mejor AUC en Test: {study.best_value:.4f}")
    
    print("\n[*] Entrenando modelo final y registrando en MLflow...")
    with mlflow.start_run(run_name="Best_XGBoost_Model"):
        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["eval_metric"] = "auc"
        
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)
        
        train_auc = roc_auc_score(y_train, final_model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
        
        mlflow.log_params(best_params)
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("test_auc", test_auc)
        
        mlflow.xgboost.log_model(final_model, "modelo_xgboost")
        print(f"[*] ¡MLflow completado con éxito! Modelo guardado correctamente.")
        return "run_id_exitoso", final_model

if __name__ == "__main__":
    run_training()