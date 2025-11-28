# Análisis de Calidad de Vino Tinto mediante Técnicas de Aprendizaje Automático Supervisado y No Supervisado

**Autores:** [Tu Nombre]  
**Fecha:** Noviembre 2025  
**Resumen:** Este estudio aplica técnicas de Machine Learning al conjunto de datos "Red Wine Quality" para analizar y predecir la calidad del vino. Se implementaron tres fases: análisis exploratorio no supervisado (PCA y K-Means) para identificar patrones latentes; modelos de regresión (lineal, polinomial y gradiente descendente) para predecir puntajes de calidad; y algoritmos de clasificación (Naive Bayes, KNN, Perceptrón) para categorizar vinos como "Buenos" o "Malos". Los resultados demuestran que el alcohol y la acidez volátil son determinantes clave, y que los modelos no lineales superan a los lineales en precisión predictiva.

---

## 1. Introducción
La industria vitivinícola depende cada vez más del análisis de datos para asegurar la calidad del producto. Este proyecto tiene como objetivo desarrollar modelos predictivos que permitan estimar la calidad del vino tinto basándose únicamente en sus propiedades fisicoquímicas, reduciendo la dependencia de catas subjetivas.

## 2. Metodología

### 2.1 Conjunto de Datos
Se utilizó el dataset "Red Wine Quality" (Cortez et al., 2009), que contiene 1599 muestras con 11 variables fisicoquímicas (acidez, azúcar, pH, alcohol, etc.) y una variable objetivo de calidad (puntaje 0-10).

### 2.2 Fase 1: Análisis No Supervisado
Antes de la predicción, se exploró la estructura de los datos:
- **PCA (Análisis de Componentes Principales):** Se redujo la dimensionalidad para visualizar la varianza explicada. Se observó que las dos primeras componentes capturan una parte significativa de la variabilidad, permitiendo visualizar la separación de calidades.
- **K-Means Clustering:** Se agruparon los vinos en clusters (k=3) basados puramente en química. Se compararon estos clusters con las etiquetas reales de calidad para evaluar si la química por sí sola define grupos naturales de calidad.

### 2.3 Fase 2: Regresión (Predicción de Puntaje)
Se buscó predecir el puntaje exacto de calidad:
- **Selección de Características:** Se utilizaron métodos *Forward* y *Backward Selection* para identificar las variables más influyentes. El alcohol emergió consistentemente como el predictor individual más fuerte.
- **Regresión Lineal vs. Polinomial:** Se comparó un ajuste lineal simple con uno polinomial de grado 2. El modelo polinomial mostró un mejor ajuste ($R^2$ superior), indicando que la relación entre químicos y calidad no es estrictamente lineal.
- **Gradiente Descendente:** Se implementó regresión múltiple optimizada mediante gradiente descendente, visualizando la convergencia de la función de costo (MSE) a través de las iteraciones.

### 2.4 Fase 3: Clasificación (Bueno vs. Malo)
Se transformó el problema en una clasificación binaria (Calidad $\ge$ 6 = Bueno):
- **Naive Bayes:** Utilizado como línea base, asumiendo independencia entre variables.
- **KNN (K-Vecinos Más Cercanos):** Se aplicó con datos normalizados (StandardScaler) para evitar sesgos por escalas de magnitud.
- **Perceptrón:** Se evaluó como clasificador lineal. Las dificultades de convergencia confirmaron que las clases "Bueno" y "Malo" no son linealmente separables en el espacio de características original.

## 3. Resultados y Discusión

### 3.1 Análisis Exploratorio
El PCA reveló que la calidad del vino no está determinada por una sola dimensión, sino por una interacción compleja de factores. Los clusters de K-Means mostraron cierta superposición con los niveles de calidad, pero no una correspondencia 1 a 1, sugiriendo que la calidad es un espectro continuo.

### 3.2 Predicción de Calidad (Regresión)
- El modelo polinomial superó al lineal, capturando mejor las relaciones no lineales en los extremos de calidad.
- El algoritmo de Gradiente Descendente convergió exitosamente, minimizando el error cuadrático medio (MSE) de manera efectiva.

### 3.3 Clasificación Binaria
- **KNN** demostró ser el modelo más robusto, adaptándose mejor a las fronteras de decisión no lineales.
- **Naive Bayes** ofreció un rendimiento aceptable pero inferior a KNN, probablemente debido a la correlación entre variables químicas que viola su asunción de independencia.
- **Perceptrón** tuvo el desempeño más bajo, oscilando debido a la no separabilidad lineal de los datos.

## 4. Conclusiones
1. **Importancia de Variables:** El nivel de alcohol y la acidez volátil son los indicadores químicos más fuertes de la calidad.
2. **Complejidad del Modelo:** Los modelos no lineales (Polinomial, KNN) superan consistentemente a los lineales, reflejando la complejidad química del vino.
3. **Aplicación Práctica:** La herramienta desarrollada permite una evaluación preliminar rápida y objetiva de la calidad, útil para el control de procesos en bodegas.

## 5. Referencias
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
