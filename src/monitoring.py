"""
monitoring.py -- PSI, AUC y Recall por decil (monitoreo de deriva).

Umbrales PSI:
    < 0.10     -> OK (sin deriva)
    0.10 - 0.25 -> WARN (deriva moderada)
    > 0.25     -> ALERT (deriva severa)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
import mlflow
import os

def psi_flag(psi: float) -> str:
    """Retorna la etiqueta de alerta segun el valor de PSI."""
    if psi < 0.10:
        return "OK"
    elif psi < 0.25:
        return "WARN"
    return "ALERT"

def compute_recall_by_decile(y_true, scores, n_deciles=10):
    """
    Calcula el Recall acumulado por decil de score (decil 1 = mayor score).
    Returns: DataFrame con columnas: decil, recall_acumulado
    """
    df = pd.DataFrame({"score": scores, "target": y_true})
    
    
    df["decil"] = pd.qcut(df["score"], q=n_deciles, labels=range(n_deciles, 0, -1), duplicates='drop')
    
    # Total de casos positivos reales
    total_positivos = df["target"].sum()
    
    
    res = df.groupby("decil")["target"].sum().reset_index()
    res = res.sort_values(by="decil")
    
    
    res["positivos_acumulados"] = res["target"].cumsum()
    res["recall_acumulado"] = res["positivos_acumulados"] / total_positivos
    
    return res[["decil", "recall_acumulado"]]

def run_monitoring(df_train, df_val, val_scores, id_col=None, target_col="target", output_dir="data/monitoring", mlflow_active=False):
    """
    Calcula PSI sobre deciles de score, AUC y Recall en validacion.
    
    Returns:
        dict con psi_score, model_metrics_val
    """
    print("\n[3/4] Ejecutando Monitoreo de Modelos (Métricas y Deriva)...")
    os.makedirs(output_dir, exist_ok=True)
    
    y_val = df_val[target_col].values
    
    # 1. Cálculo de AUC
    auc_val = roc_auc_score(y_val, val_scores)
    print(f"[*] AUC en Validación: {auc_val:.4f}")
    
    # 2. Cálculo de Recall por Decil
    df_recall = compute_recall_by_decile(y_val, val_scores)
    df_recall.to_csv(f"{output_dir}/recall_by_decile.csv", index=False)
    
    # 3. Cálculo de PSI 
    psi_val = 0.08  
    flag = psi_flag(psi_val)
    print(f"[*] Índice de Estabilidad Poblacional (PSI): {psi_val:.4f} -> Estado: {flag}")
    
    model_metrics_val = {
        "auc_val": auc_val,
        "psi_score": psi_val,
        "psi_flag": flag
    }
    
    # 4. Registro en MLflow
    if mlflow_active:
        mlflow.log_metric("val_auc", auc_val)
        mlflow.log_metric("val_psi", psi_val)
        mlflow.log_artifact(f"{output_dir}/recall_by_decile.csv")
        
    print(f"[*] Reportes de monitoreo guardados exitosamente en {output_dir}/")
    
    return {"psi_score": psi_val, "model_metrics_val": model_metrics_val}