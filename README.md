# Sistema de Predicción Temprana de Deserción y Éxito Académico (Spark ML)

**Autor:** Andrés Felipe Maestre Badillo  
**ID:** U00071243  

## Descripción
Aplicación web interactiva desarrollada en **Streamlit** que despliega un `PipelineModel` (`RandomForestClassifier`) entrenado en **Apache Spark ML** para clasificar estudiantes universitarios en tres categorías: `Graduate`, `Dropout` y `Enrolled`.

## Cómo ejecutarla
1. Asegurarse de contar con Java (`default-jre`) y Python 3.10+.
2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Iniciar la aplicación:
   ```bash
   streamlit run app.py
   ```

## URL de la aplicación desplegada
- **Streamlit Community Cloud:** [https://parcial-sparkml-app-sjimnhcnsezdsqwri9qwy4.streamlit.app/]