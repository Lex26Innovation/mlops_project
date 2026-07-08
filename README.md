
**Autor:** [ALEX GENARO TTITO TORRES]
**Link del Repositorio GitHub: https://github.com/Lex26Innovation/mlops_project

## Estructura
- `data/`: Contiene los datos crudos, procesados, métricas de monitoreo y réplicas.
- `mlruns/`: Contiene el tracking de MLflow y los modelos guardados.
- `src/`: Scripts de preprocesamiento, entrenamiento, inferencia, etc.
- `main.py`: Orquestador End-to-End.
- `dashboard.py`: Dashboard interactivo en Streamlit para visualizar resultados.
- `requirements.txt`: Dependencias del proyecto.

## Instrucciones de Instalación
1. Crear un entorno virtual (Python 3.8+).
2. Instalar las dependencias ejecutando:
   ```bash
   pip install -r requirements.txt

## Instrucciones de Ejecución
1. Para la data Out-of-Time (OOT), Colocar la data nueva en la carpeta correspondiente y ejecutar: python src/inference.py
2. Para reentrenar el modelo (con búsqueda de hiperparámetros en Optuna): python main.py
3. Para visualizar los resultados (Dashboard): streamlit run dashboard.py