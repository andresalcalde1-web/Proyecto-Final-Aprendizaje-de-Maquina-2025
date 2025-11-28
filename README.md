# Análisis de Calidad del Vino Tinto

Proyecto de Machine Learning para analizar y predecir la calidad del vino tinto mediante técnicas de Aprendizaje Supervisado y No Supervisado.

## Requisitos

- Python 3.11 o superior
- pip (gestor de paquetes de Python)

## Instalación

1. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

### Ejecutar la Aplicación Streamlit (Recomendado)

La forma más sencilla de explorar el proyecto es mediante la interfaz web interactiva:

```bash
streamlit run app.py
```

Esto abrirá automáticamente una ventana del navegador con la aplicación. Si no se abre automáticamente, visita: `http://localhost:8501`

## Estructura del Proyecto

```
red_wine_quality/
│
├── app.py                          # Aplicación Streamlit principal (EJECUTAR ESTE)
├── data_loader.py                  # Script para descargar y preprocesar datos
├── analysis_unsupervised.py        # Análisis PCA y K-Means (ya ejecutado)
├── analysis_regression.py          # Análisis de regresión (ya ejecutado)
├── analysis_classification.py      # Análisis de clasificación (ya ejecutado)
├── processed_red_wine.csv          # Dataset preprocesado
├── Final_Report.md                 # Reporte técnico en formato académico
├── requirements.txt                # Dependencias del proyecto
└── *.png                           # Gráficas generadas por los análisis
```

## Funcionalidades de la Aplicación

La aplicación Streamlit incluye 5 secciones principales:

1. **Explorador de Datos**: Visualiza el dataset completo (1599 muestras)
2. **Aprendizaje No Supervisado**: PCA y K-Means Clustering
3. **Regresión**: Predicción de puntaje de calidad (0-10)
4. **Clasificación**: Clasificación binaria Bueno vs Malo (Naive Bayes, KNN, Perceptrón)
5. **Herramienta de Predicción**: Predicción individual con verificación histórica

## Dataset

El proyecto utiliza el dataset "Red Wine Quality" de UCI Machine Learning Repository, que contiene 1599 muestras de vino tinto portugués con 11 características fisicoquímicas:

- Acidez fija
- Acidez volátil
- Ácido cítrico
- Azúcar residual
- Cloruros
- Dióxido de azufre libre
- Dióxido de azufre total
- Densidad
- pH
- Sulfatos
- Alcohol

## Notas Importantes

- **NO es necesario ejecutar** los scripts `analysis_*.py` individualmente. Ya se ejecutaron y generaron los resultados.
- El archivo `processed_red_wine.csv` ya está incluido, por lo que tampoco es necesario ejecutar `data_loader.py`.
- **Solo ejecute `streamlit run app.py`** para explorar todo el proyecto.

## Tecnologías Utilizadas

- **Python 3.11**
- **Streamlit**: Framework para aplicaciones web
- **Scikit-learn**: Algoritmos de Machine Learning
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **Matplotlib & Seaborn**: Visualización de datos

## Autor

Andrés, Daniela
Universidad Tecnológica de Pereira
# Proyecto-Final-Aprendizaje-de-Maquina-2025
