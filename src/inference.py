import os
import pandas as pd
import mlflow
from pathlib import Path
import importlib.abc
import importlib.resources.abc
importlib.abc.Traversable = importlib.resources.abc.Traversable

MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#  Run ID de MLflow
RUN_ID = "c9e73051f571466eb0311551661b80bc" 
#  ruta real de Windows
ruta_local = "mlruns/1/models/m-3d6f16dc8e2e4fe6b11f97838696a971/artifacts"

MODEL_URI = Path(ruta_local).resolve().as_uri()

def run_inference():
    print("\n" + "="*50)
    print("[*] INICIANDO PIPELINE DE INFERENCIA")
    print("="*50 + "\n")
    
    
    print(f"[*] Descargando y cargando el modelo campeón desde MLflow (Run ID: {RUN_ID})...")
    try:
        model = mlflow.xgboost.load_model(MODEL_URI)
        print("[✓] Modelo cargado exitosamente.")
    except Exception as e:
        print(f"[X] Error al cargar el modelo: {e}")
        return

    #  Carga los datos que queremos predecir ( df_test)
    print("[*] Cargando datos de clientes para inferencia...")
    if not os.path.exists("data/processed/df_test.csv"):
        print("[X] Error: No se encontró el archivo data/processed/df_test.csv")
        return
        
    df_nuevos_clientes = pd.read_csv("data/processed/df_test.csv")
    
    
    id_clientes = df_nuevos_clientes["key_value"].copy() if "key_value" in df_nuevos_clientes.columns else range(len(df_nuevos_clientes))

    
    cols_to_drop = ["target", "p_fecinformacion", "key_value", "monto", "grp_campecs06m", "prob_value_con"]
    
    
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_nuevos_clientes.columns]
    X_inferencia = df_nuevos_clientes.drop(columns=existing_cols_to_drop)

    
    print("[*] Ejecutando el modelo sobre los datos de los clientes...")
    
    if 'prob_value_contact' in X_inferencia.columns:
        X_inferencia = X_inferencia.drop(columns=['prob_value_contact'])
    
    columnas_del_modelo = [
        'flag_camp_sms', 'flag_camp_email', 'flag_camp_wsp', 'flag_camp_call', 'monto', 
        'sub_segmento_G1', 'sub_segmento_G2', 'sub_segmento_G3', 'sub_segmento_G4', 'sub_segmento_G5', 
        'grp_campecs06m_G1', 'grp_campecs06m_G2', 'grp_campecs06m_G3', 'grp_campecs06m_G4', 'grp_campecs06m_G5'
    ]
    
    
    
    X_solo_features = X_inferencia[columnas_del_modelo]
    
    
    print("[*] Generando predicciones con las características alineadas...")
    predicciones_prob = model.predict_proba(X_solo_features)[:, 1]

    # Estructura del resultado final (Postprocessing)
    print("[*] Formateando resultados finales...")
    df_resultados = pd.DataFrame({
        "id_cliente": id_clientes,
        "score_prediccion": predicciones_prob
    })

    # Guardado los resultados en la carpeta postprocessed
    os.makedirs("data/postprocessed", exist_ok=True)
    output_path = "data/postprocessed/predicciones_finales.csv"
    df_resultados.to_csv(output_path, index=False)
    
    print(f"\n[✓] ¡Pipeline completado! Resultados guardados en: {output_path}")
    print(f"[*] Se generaron predicciones para {len(df_resultados)} clientes.\n")
    return predicciones_prob

if __name__ == "__main__":
    run_inference()