# Sistema de Predicción Temprana de Deserción y Éxito Académico (Spark ML)

**Autor:** Andrés Felipe Maestre Badillo  
**ID:** U00071243  

## Descripción
Aplicación web interactiva desarrollada en **Streamlit** que despliega un `PipelineModel` (`RandomForestClassifier`) entrenado en **Apache Spark ML** para clasificar estudiantes universitarios en tres categorías: `Graduate`, `Dropout` y `Enrolled`.

El flujo completo incluye:
- Preprocesamiento y limpieza de datos en PySpark (`Parcial_BIGDATA.ipynb`).
- Ingeniería de características (tasas de aprobación del 1.er y 2.º semestre y variación de notas).
- Pipeline completo (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler` y clasificación multiclase con ajuste de hiperparámetros por validación cruzada).
- Despliegue del modelo entrenado (`modelo_final/`) en una interfaz web con Streamlit.

## Cómo ejecutarla localmente
1. Asegurarse de contar con Java (`default-jre`) y Python 3.10+.
2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Iniciar la aplicación:
   ```bash
   streamlit run app.py
   ```

## Repositorio y URL de la aplicación desplegada
- **Repositorio de GitHub:** https://github.com/afmae556/parcial-sparkml-streamlit
- **Streamlit Community Cloud:** *(Pendiente de pegar la URL pública tras el despliegue en share.streamlit.io)*
