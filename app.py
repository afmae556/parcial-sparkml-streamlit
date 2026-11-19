import json
import os
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Alerta Temprana de Deserción Estudiantil - Spark ML",
    page_icon="🎓",
    layout="wide",
)

# Requisito de diseño: Estilo propio y coherente (colores, tipografía y tarjetas)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .titulo-app {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitulo-app {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .caja-resultado {
        padding: 1.4rem;
        border-radius: 12px;
        background-color: #EFF6FF;
        border-left: 6px solid #2563EB;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .autor-footer {
        margin-top: 3rem;
        padding: 1rem;
        border-top: 2px solid #E2E8F0;
        text-align: center;
        color: #334155;
        font-size: 0.95rem;
        background-color: #F8FAFC;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Requisito de diseño: Al menos una imagen (logo o banner)
if os.path.exists("banner.png"):
  st.image("banner.png", use_container_width=True)
elif os.path.exists("banner.jpg"):
  st.image("banner.jpg", use_container_width=True)
else:
  st.image(
      "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?auto=format&fit=crop&w=1200&q=80",
      use_container_width=True,
  )

# Requisito de diseño: Título claro
st.markdown(
    '<div class="titulo-app">🎓 Sistema de Predicción Temprana de Deserción y'
    " Éxito Académico</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitulo-app">Modelo RandomForestClassifier entrenado con'
    " Apache Spark ML</div>",
    unsafe_allow_html=True,
)

# Requisito de diseño: Instrucciones de uso visibles
st.info("""
    **📌 Instrucciones de uso:**
    1. Complete los datos socioeconómicos y las calificaciones del **1.er y 2.º semestre** en el formulario inferior.
    2. Presione el botón **Predecir**.
    3. La aplicación calculará automáticamente las **3 variables derivadas** (tasas de aprobación del 1.er y 2.º semestre y variación de la nota), ejecutará el `PipelineModel` de Spark ML y mostrará la clase predicha junto con las probabilidades por categoría.
    """)


# Punto 7(a): Cargar Spark y el mejor modelo con PipelineModel.load y @st.cache_resource
@st.cache_resource
def iniciar_spark_y_modelo():
  spark = (
      SparkSession.builder.appName("dropout_streamlit_sparkml")
      .master("local[*]")
      .config("spark.driver.memory", "2g")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.constraintPropagation.enabled", "false")
      .getOrCreate()
  )
  modelo = PipelineModel.load("modelo_final")

  if os.path.exists("labels.json"):
    with open("labels.json", "r", encoding="utf-8") as f:
      labels = json.load(f)
  else:
    labels = ["Graduate", "Dropout", "Enrolled"]

  return spark, modelo, labels


spark, modelo_final, labels_orden = iniciar_spark_y_modelo()

# Punto 7(b): Formulario (st.form) con todas las variables que usa el modelo y botón Predecir
with st.form("form_prediccion"):
  st.subheader("📋 Formulario de Evaluación del Estudiante")
  col1, col2, col3 = st.columns(3)

  with col1:
    st.markdown("**1. Datos Socioeconómicos**")
    age_at_enrollment = st.number_input(
        "Edad al inscribirse", min_value=16, max_value=70, value=20
    )
    gender = st.selectbox(
        "Género",
        options=[0, 1],
        format_func=lambda x: "Femenino (0)" if x == 0 else "Masculino (1)",
    )
    displaced = st.selectbox(
        "¿Desplazado / fuera de su ciudad?",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Sí (1)",
    )
    scholarship_holder = st.selectbox(
        "¿Posee beca?",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Sí (1)",
    )
    debtor = st.selectbox(
        "¿Tiene deudas con la universidad?",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Sí (1)",
    )
    tuition_fees_up_to_date = st.selectbox(
        "¿Matrícula al día?",
        options=[1, 0],
        format_func=lambda x: "Sí (1)" if x == 1 else "No (0)",
    )

  with col2:
    st.markdown("**2. Rendimiento 1.er Semestre**")
    sem1_enrolled = st.number_input(
        "Unidades inscritas (1.er Sem)", min_value=0, max_value=26, value=6
    )
    sem1_approved = st.number_input(
        "Unidades aprobadas (1.er Sem)", min_value=0, max_value=26, value=5
    )
    sem1_grade = st.number_input(
        "Nota promedio (1.er Sem, escala 0 a 20)",
        min_value=0.0,
        max_value=20.0,
        value=13.0,
        step=0.1,
    )

  with col3:
    st.markdown("**3. Rendimiento 2.º Semestre**")
    sem2_enrolled = st.number_input(
        "Unidades inscritas (2.º Sem)", min_value=0, max_value=26, value=6
    )
    sem2_approved = st.number_input(
        "Unidades aprobadas (2.º Sem)", min_value=0, max_value=26, value=5
    )
    sem2_grade = st.number_input(
        "Nota promedio (2.º Sem, escala 0 a 20)",
        min_value=0.0,
        max_value=20.0,
        value=13.5,
        step=0.1,
    )

  predecir_btn = st.form_submit_button("🔍 Predecir", use_container_width=True)

if predecir_btn:
  # Punto 7(d): Convertir los datos en un DataFrame de Spark
  fila_usuario = [{
      "tuition_fees_up_to_date": int(tuition_fees_up_to_date),
      "scholarship_holder": int(scholarship_holder),
      "debtor": int(debtor),
      "gender": int(gender),
      "displaced": int(displaced),
      "age_at_enrollment": float(age_at_enrollment),
      "curricular_units_1st_sem_enrolled": float(sem1_enrolled),
      "curricular_units_1st_sem_approved": float(sem1_approved),
      "curricular_units_1st_sem_grade": float(sem1_grade),
      "curricular_units_2nd_sem_enrolled": float(sem2_enrolled),
      "curricular_units_2nd_sem_approved": float(sem2_approved),
      "curricular_units_2nd_sem_grade": float(sem2_grade),
  }]

  df_input = spark.createDataFrame(fila_usuario)

  # Punto 7(c): Calcular dentro de la app las 3 variables derivadas del Punto 3
  df_input = (
      df_input.withColumn(
          "tasa_aprobacion_sem1",
          F.when(
              F.col("curricular_units_1st_sem_enrolled") > 0,
              F.col("curricular_units_1st_sem_approved")
              / F.col("curricular_units_1st_sem_enrolled"),
          ).otherwise(0.0),
      )
      .withColumn(
          "tasa_aprobacion_sem2",
          F.when(
              F.col("curricular_units_2nd_sem_enrolled") > 0,
              F.col("curricular_units_2nd_sem_approved")
              / F.col("curricular_units_2nd_sem_enrolled"),
          ).otherwise(0.0),
      )
      .withColumn(
          "variacion_nota_semestres",
          F.col("curricular_units_2nd_sem_grade")
          - F.col("curricular_units_1st_sem_grade"),
      )
  )

  # Aplicar el PipelineModel
  prediccion_sdf = modelo_final.transform(df_input)
  resultado = prediccion_sdf.select(
      "prediction",
      "probability",
      "tasa_aprobacion_sem1",
      "tasa_aprobacion_sem2",
      "variacion_nota_semestres",
  ).collect()[0]

  # Punto 7(e): Traducir con el mismo orden de labels del StringIndexer
  idx_predicho = int(resultado["prediction"])
  clase_predicha = labels_orden[idx_predicho]
  vector_prob = resultado["probability"].toArray()[: len(labels_orden)]

  descripcion_clase = {
      "Graduate": "Graduate — Camino a Graduarse 🎓",
      "Dropout": "Dropout — En Riesgo de Deserción ⚠️",
      "Enrolled": "Enrolled — Continúa Matriculado 📚",
  }

  st.markdown(
      f"""
        <div class="caja-resultado">
            <h2>Clase Predicha: <strong>{descripcion_clase.get(clase_predicha, clase_predicha)}</strong></h2>
            <p><b>Variables derivadas calculadas:</b> Tasa de aprobación 1.er Sem = <code>{resultado['tasa_aprobacion_sem1']*100:.1f}%</code> |
            Tasa de aprobación 2.º Sem = <code>{resultado['tasa_aprobacion_sem2']*100:.1f}%</code> |
            Variación de nota = <code>{resultado['variacion_nota_semestres']:+.2f}</code></p>
        </div>
        """,
      unsafe_allow_html=True,
  )

  # Tabla y gráfico de probabilidades por cada clase
  df_probs = pd.DataFrame({
      "Clase": labels_orden,
      "Probabilidad": [round(float(p), 4) for p in vector_prob],
      "Porcentaje (%)": [round(float(p) * 100, 2) for p in vector_prob],
  })

  col_t, col_g = st.columns([1, 1.4])
  with col_t:
    st.markdown("#### 📊 Tabla de Probabilidades por Clase")
    st.dataframe(df_probs, use_container_width=True, hide_index=True)
  with col_g:
    st.markdown("#### 📈 Gráfico de Probabilidades por Clase")
    st.bar_chart(df_probs.set_index("Clase")["Porcentaje (%)"])

# Requisito de diseño: Al final de la página, el nombre del estudiante como autor
st.markdown(
    '<div class="autor-footer">Autor: <strong>Andrés Felipe Maestre Badillo</strong> |'
    " ID: <strong>U00071243</strong> — Examen de Big Data (Spark ML)</div>",
    unsafe_allow_html=True,
)