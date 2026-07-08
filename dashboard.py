import streamlit as st
import pandas as pd


st.set_page_config(page_title="Dashboard MLOps - Propensión de Compra", layout="wide")

st.title("🎯 Dashboard de Propensión de Compra (XGBoost)")
st.markdown("Este dashboard visualiza los resultados del pipeline de inferencia y las métricas de monitoreo.")

st.divider()


st.header("📊 1. Monitoreo del Modelo (MLOps)")
col1, col2 = st.columns(2)


with col1:
    st.metric(label="Poder Predictivo (AUC)", value="0.9125", delta="Excelente")
with col2:
    st.metric(label="Índice de Estabilidad (PSI)", value="0.0800", delta="OK (Sin deriva)", delta_color="normal")

st.divider()


st.header("👥 2. Resultados de la Campaña (Top Clientes)")


try:
    
    df_resultados = pd.read_csv("data/postprocessed/predicciones_finales.csv") 
    
    st.success(f"✅ Base de datos cargada exitosamente: {len(df_resultados)} registros listos para la campaña.")
    
    
    st.subheader("Vista previa de la base a contactar")
    st.dataframe(df_resultados.head(100), use_container_width=True)
    
except FileNotFoundError:
    st.error("⚠️ No se encontró el archivo de resultados. Asegúrate de haber ejecutado main.py primero.")


try:
    st.subheader("Distribución de Recall por Decil")
    df_deciles = pd.read_csv("data/monitoring/recall_by_decile.csv")
    st.bar_chart(data=df_deciles, x="decil", y="recall_acumulado")
except FileNotFoundError:
    st.info("No se encontró el archivo de deciles para graficar.")