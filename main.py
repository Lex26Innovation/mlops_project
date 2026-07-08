"""
main.py -- Orquestador del pipeline ML E2E.
"""
import numpy as np
import pandas as pd
from src.preprocessing import run_preprocessing
from src.entrenamiento import train_and_log
from src.postprocessing import run_postprocessing, save_replica
from src.monitoring import run_monitoring


# from src.monitoring import run_monitoring, compute_recall_by_decile

INPUT_PATH = "data/raw" 
OUTPUT_DIR = "data/processed"
POST_PATH = "data/postprocessed/output_tlv.csv"

def main():
    # 
    print("\n[1/4] Ejecutando Pipeline de Preprocesamiento...")
    df_train, df_test, df_val, meta = run_preprocessing(INPUT_PATH)
    
    # #
    print("\n[2/4] Iniciando Entrenamiento y Optimización con Optuna...")
    run_id, model = train_and_log(
        train_path = OUTPUT_DIR + "/df_train.csv",
        test_path = OUTPUT_DIR + "/df_test.csv",
        val_path = OUTPUT_DIR + "/df_val.csv"
    )
    
    # 
    print("\n[3/4] Generando Scores de validación para Monitoreo...")
    
    
 #   columnas_del_modelo = [
 #       'flag_camp_sms', 'flag_camp_email', 'flag_camp_wsp', 'flag_camp_call', 'monto', 
 #      'sub_segmento_G1', 'sub_segmento_G2', 'sub_segmento_G3', 'sub_segmento_G4', 'sub_segmento_G5', 
 #      'grp_campecs06m_G1', 'grp_campecs06m_G2', 'grp_campecs06m_G3', 'grp_campecs06m_G4', 'grp_campecs06m_G5'
 #  ]
 # 
    columnas_reales = model.feature_names_in_

    
    X_val = df_test[columnas_reales]
    val_scores = model.predict_proba(X_val)[:, 1]
    
    
    run_monitoring(df_train, df_test, val_scores)
    # compute_recall_by_decile(df_val["target"], val_scores)
    
    # 
    print("\n[4/4] Ejecutando Post-procesamiento (Scoring TLV) y Réplicas...")
    df_resultado = run_postprocessing(val_scores, df_test, POST_PATH)
    
    save_replica(df_resultado, table="EC_OMNICANAL", partition="202412")
    
    print("\n🎉 ¡PIPELINE END-TO-END EJECUTADO EXITOSAMENTE! 🎉\n")

if __name__ == "__main__":
    main()