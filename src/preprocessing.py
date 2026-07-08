"""
preprocessing.py -- Limpieza y transformacion del dataset CU Venta.
Produce: df_train.csv, df_test.csv, df_val.csv
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

NAN_THRESHOLD = 80
VALIDATION_CODMES = 201912.0
TEST_SIZE = 0.30
RANDOM_STATE = 123

def run_preprocessing(data_dir="data/raw", nan_threshold=NAN_THRESHOLD):
    print(f"[*] Buscando particiones CSV en: {data_dir}")
    
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        raise ValueError(f"No se encontraron archivos en: {data_dir}. ¿Los descargaste?")
        
    print(f"[*] Concatenando {len(all_files)} archivos...")
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    print(f"[*] Dataset original: {df.shape}")

    # Eliminar columnas con exceso de NaN
    cols_drop = [c for c in df.columns if df[c].isna().mean() * 100 > nan_threshold]
    df = df.drop(columns=cols_drop)
    
    # Imputaciones y encodings
    for col in df.columns:
        
        if col in ["p_fecinformacion", "key_value", "target", "monto", "grp_campecs06m", "prob_value_contact"]:
            if df[col].isna().sum() > 0 and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            continue
            
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Desconocido'
            df[col] = df[col].fillna(mode_val)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    
    df_val = df[df["p_fecinformacion"] == VALIDATION_CODMES].copy()
    df_main = df[df["p_fecinformacion"] != VALIDATION_CODMES].copy()
    
    df_train, df_test = train_test_split(
        df_main, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"[*] Splits terminados -> Train: {df_train.shape[0]} | Test: {df_test.shape[0]} | Val: {df_val.shape[0]}")
    
    # Guardar localmente
    df_train.to_csv("data/processed/df_train.csv", index=False)
    df_test.to_csv("data/processed/df_test.csv", index=False)
    df_val.to_csv("data/processed/df_val.csv", index=False)
    
    return df_train, df_test, df_val, {"dropped": cols_drop}

if __name__ == "__main__":
    run_preprocessing()