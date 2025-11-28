import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, roc_curve, auc, classification_report





# Page Config
st.set_page_config(page_title="Análisis de Calidad del Vino Tinto", layout="wide")

st.title("Análisis de Calidad del Vino Tinto")
st.markdown("Análisis integral de la calidad del vino tinto mediante Aprendizaje No Supervisado, Regresión y Clasificación.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("processed_red_wine.csv")
    
    # Traducir nombres de columnas a español
    column_translation = {
        'fixed acidity': 'acidez_fija',
        'volatile acidity': 'acidez_volatil',
        'citric acid': 'acido_citrico',
        'residual sugar': 'azucar_residual',
        'chlorides': 'cloruros',
        'free sulfur dioxide': 'dioxido_azufre_libre',
        'total sulfur dioxide': 'dioxido_azufre_total',
        'density': 'densidad',
        'pH': 'pH',
        'sulphates': 'sulfatos',
        'alcohol': 'alcohol',
        'quality': 'calidad'
    }
    df = df.rename(columns=column_translation)
    return df

df = load_data()

# Sidebar
st.sidebar.header("Configuración")
analysis_mode = st.sidebar.radio("Seleccione la Fase de Análisis", 
                                 ["Explorador de Datos", "No Supervisado (PCA/K-Means)", "Regresión (Puntaje)", "Clasificación (Bueno/Malo)", "Herramienta de Predicción"])

# 1. Explorador de Datos
if analysis_mode == "Explorador de Datos":
    st.header("Explorador de Datos")
    
    st.markdown("""
    **Descripción:** Esta sección permite explorar el conjunto de datos de vinos tintos que contiene 1599 muestras. 
    Cada fila representa un vino diferente con sus propiedades fisicoquímicas medidas en laboratorio.
    
    **Funcionalidad:** Haga clic en los encabezados de las columnas para ordenar los datos por esa característica.
    """)
    
    # Mostrar todos los datos con tabla scrollable
    st.dataframe(df, height=600, width='stretch')
    
    st.subheader("Matriz de Correlación")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.subheader("Distribución de Características")
    feature = st.selectbox("Seleccione una Característica", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

# 2. Unsupervised Analysis
elif analysis_mode == "No Supervisado (PCA/K-Means)":
    st.header("Aprendizaje No Supervisado")
    
    st.markdown("""
    **Descripción:** El aprendizaje no supervisado busca patrones en los datos sin usar las etiquetas de calidad.
    
    - **PCA (Análisis de Componentes Principales):** Reduce las 11 variables químicas a 2 dimensiones principales 
      para visualizar la estructura de los datos. La "varianza explicada" indica cuánta información se conserva.
    
    - **K-Means:** Agrupa automáticamente los vinos en clusters basándose solo en su química, 
      sin conocer su calidad real. Permite identificar grupos naturales de vinos similares.
    """)
    
    # Prepare Data
    X = df.drop('calidad', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PCA (Reducción de Dimensionalidad)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        explained_variance = pca.explained_variance_ratio_
        st.info(f"Varianza Explicada: PC1 ({explained_variance[0]:.2%}), PC2 ({explained_variance[1]:.2%})")
        
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['calidad'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Calidad')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA de Calidad del Vino Tinto')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Clustering K-Means")
        k = st.slider("Número de Clusters (k)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Clustering K-Means (k={k})')
        st.pyplot(fig)

# 3. Regression Analysis
elif analysis_mode == "Regresión (Puntaje)":
    st.header("Análisis de Regresión (Predicción de Puntaje 0-10)")
    
    st.markdown("""
    **Descripción:** La regresión predice un valor numérico continuo (puntaje de calidad 0-10) 
    basándose en las características químicas del vino.
    
    - **Regresión Lineal:** Asume una relación lineal directa entre la variable química y la calidad.
    - **Regresión Polinomial:** Permite capturar relaciones no lineales (curvas).
    - **R² (Coeficiente de Determinación):** Mide qué tan bien el modelo explica la variabilidad de los datos. 
      Valores cercanos a 1.0 indican mejor ajuste.
    - **Descenso de Gradiente:** Algoritmo de optimización que minimiza el error del modelo ajustando 
      iterativamente los pesos de todas las variables.
    """)
    
    X = df.drop('calidad', axis=1)
    y = df['calidad']
    
    # --- Feature Importance Analysis ---
    st.subheader("Análisis de Importancia de Variables")
    st.markdown("Identificación de las variables que tienen mayor impacto (correlación) con la calidad del vino.")
    
    # Calculate correlations
    correlations = df.corr()['calidad'].drop('calidad').sort_values(ascending=False)
    
    # Plot
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in correlations.values]
    sns.barplot(x=correlations.values, y=correlations.index, palette=colors, ax=ax_corr)
    ax_corr.set_title("Correlación de Características con la Calidad")
    ax_corr.set_xlabel("Coeficiente de Correlación")
    ax_corr.axvline(0, color='black', linewidth=1)
    st.pyplot(fig_corr)
    
    st.markdown("---")
    # -----------------------------------
    
    feature = st.selectbox("Seleccione una Característica para Regresión Simple", X.columns, index=10)
    
    X_single = X[[feature]]
    X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)
    
    # Linear
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    r2_lin = r2_score(y_test, y_pred_lin)
    
    # Polinomial
    degree = st.slider("Grado Polinomial", 1, 5, 2)
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_test)
    r2_poly = r2_score(y_test, y_pred_poly)
    
    st.write(f"**Linear R2:** {r2_lin:.4f}")
    st.write(f"**Polynomial (Deg {degree}) R2:** {r2_poly:.4f}")
    
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual')
    
    # Sort for plotting using Pandas (More robust)
    X_test_sorted_df = X_test.sort_values(by=feature)
    
    # Predict on sorted values to get smooth lines
    y_pred_lin_sorted = lin_reg.predict(X_test_sorted_df)
    
    X_poly_test_sorted = poly.transform(X_test_sorted_df)
    y_pred_poly_sorted = poly_reg.predict(X_poly_test_sorted)
    
    ax.plot(X_test_sorted_df[feature], y_pred_lin_sorted, color='blue', label='Linear')
    ax.plot(X_test_sorted_df[feature], y_pred_poly_sorted, color='red', linestyle='--', label=f'Poly Deg {degree}')
    
    ax.set_xlabel(feature)
    ax.set_ylabel("Calidad")
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Regresión Múltiple (Descenso de Gradiente)")
    if st.button("Ejecutar Descenso de Gradiente"):
        # CORRECCIÓN: Split ANTES del escalamiento
        X_train_gd, X_test_gd, y_train_gd, y_test_gd = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize features - FIT solo en train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_gd)
        X_test_scaled = scaler.transform(X_test_gd)
        
        # Prepare matrices for gradient descent (train only)
        m = len(y_train_gd)
        X_b = np.c_[np.ones((m, 1)), X_train_scaled]
        y_np = y_train_gd.values.reshape(-1, 1)
        
        theta = np.random.randn(X_b.shape[1], 1)
        eta = 0.1
        n_iterations = 1000
        cost_history = []
        
        progress_bar = st.progress(0)
        
        for i in range(n_iterations):
            gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y_np)
            theta = theta - eta * gradients
            cost = np.mean((X_b.dot(theta) - y_np)**2)
            cost_history.append(cost)
            
            if i % 100 == 0:
                progress_bar.progress(i / n_iterations)
                
        progress_bar.progress(1.0)
        
        # Evaluate on both train and test
        X_train_b = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
        X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]
        
        train_mse = np.mean((X_train_b.dot(theta) - y_train_gd.values.reshape(-1, 1))**2)
        test_mse = np.mean((X_test_b.dot(theta) - y_test_gd.values.reshape(-1, 1))**2)
        
        st.success(f"Descenso de Gradiente Convergió.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MSE Train", f"{train_mse:.4f}")
        with col2:
            st.metric("MSE Test", f"{test_mse:.4f}")
        
        fig, ax = plt.subplots()
        ax.plot(range(n_iterations), cost_history)
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Costo (MSE) en Train")
        ax.set_title("Función de Costo vs Iteraciones")
        st.pyplot(fig)
        plt.close(fig)

# 4. Classification Analysis
elif analysis_mode == "Clasificación (Bueno/Malo)":
    st.header("Clasificación (Vino Bueno vs Malo)")
    
    st.markdown("""
    **Descripción:** La clasificación transforma el problema en categorías binarias (Bueno/Malo) 
    en lugar de predecir un puntaje numérico.
    
    **Modelos Implementados:**
    - **Naive Bayes:** Asume independencia entre variables. Rápido y efectivo como línea base.
    - **KNN (K-Vecinos Más Cercanos):** Clasifica un vino basándose en la "mayoría de votos" de sus K vecinos más similares en el espacio químico.
    - **Perceptrón:** Clasificador lineal que traza una frontera recta entre clases. Puede tener dificultades si los datos no son linealmente separables.
    
    **Métricas:**
    - **Precisión (Accuracy):** Porcentaje de predicciones correctas.
    - **Matriz de Confusión:** Muestra verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.
    """)
    
    # Transform Target
    threshold = st.slider("Umbral de Calidad (Bueno >= X)", 3, 8, 6)
    df_copy = df.copy()
    df_copy['target'] = (df_copy['calidad'] >= threshold).astype(int)
    
    st.write(f"**Balance de Clases:** Bueno: {df_copy['target'].mean():.2%}, Malo: {(1-df_copy['target'].mean()):.2%}")
    
    X = df_copy.drop(['calidad', 'target'], axis=1)
    y = df_copy['target']
    
    # CORRECCIÓN: Split PRIMERO con ESTRATIFICACIÓN
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    model_name = st.selectbox("Seleccione el Clasificador", ["Naive Bayes", "KNN", "Perceptrón"])
    
    if model_name == "Naive Bayes":
        clf = GaussianNB()
    elif model_name == "KNN":
        k = st.slider("K Vecinos", 1, 20, 5)
        clf = KNeighborsClassifier(n_neighbors=k)
    else:
        clf = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.metric("Precisión (Accuracy)", f"{acc:.4f}")
    
    # Classification Report (Translated)
    st.subheader("Reporte de Clasificación Detallado")
    report_dict = classification_report(y_test, y_pred, target_names=['Malo', 'Bueno'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Translate columns and index
    report_df.columns = ['Precisión', 'Sensibilidad (Recall)', 'Puntaje F1 (F1-Score)', 'Soporte']
    report_df.index = ['Malo', 'Bueno', 'Accuracy', 'Promedio Macro', 'Promedio Ponderado']
    
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
    
    with st.expander("¿Qué significan estas métricas?"):
        st.markdown("""
        - **Precisión:** De todos los vinos que el modelo predijo como "Buenos", ¿cuántos eran realmente buenos? (Evita falsas alarmas).
        - **Sensibilidad (Recall):** De todos los vinos que eran realmente "Buenos", ¿cuántos detectó el modelo? (Evita perder oportunidades).
        - **Puntaje F1:** Es el balance entre Precisión y Sensibilidad. Es la mejor métrica general si las clases están desbalanceadas.
        - **Soporte:** Cantidad de vinos reales de cada clase en el test set.
        """)
    
    # Calculate ROC/AUC
    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_prob = clf.decision_function(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malo', 'Bueno'], yticklabels=['Malo', 'Bueno'], ax=ax)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.subheader("Curva ROC y AUC")
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            st.metric("AUC Score", f"{roc_auc:.4f}")
            
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('Tasa de Falsos Positivos')
            ax_roc.set_ylabel('Tasa de Verdaderos Positivos')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        else:
            st.warning("Este modelo no soporta cálculo de probabilidades para curva ROC.")

        if model_name == "Perceptrón":
            st.warning("Nota: Para el Perceptrón se usa la función de decisión ya que no produce probabilidades directas.")
        


# 5. Prediction Tool
elif analysis_mode == "Herramienta de Predicción":
    st.header("Herramienta de Predicción Individual")
    
    st.markdown("""
    **Descripción:** Esta herramienta permite ingresar las propiedades químicas de un vino hipotético 
    y obtener dos predicciones:
    
    1. **Puntaje (Regresión):** Estimación numérica de la calidad (0-10) usando Regresión Lineal.
    2. **Clasificación (KNN):** Categorización como "Bueno" o "Malo" basada en los 5 vecinos más cercanos.
    
    **Comparativa de Referencia:** Compara las propiedades ingresadas con el promedio de vinos buenos (calidad ≥ 6) 
    del dataset para contextualizar la predicción.
    """)
    
    # Train Models on Full Data
    X = df.drop('calidad', axis=1)
    y_score = df['calidad']
    y_class = (df['calidad'] >= 6).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Regressor (for Score)
    reg = LinearRegression()
    reg.fit(X_scaled, y_score)
    
    # Classifier (for Good/Bad) - Using KNN as it performed well
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_scaled, y_class)
    
    # Input Form
    col1, col2, col3 = st.columns(3)
    
    inputs = {}
    features = X.columns.tolist()
    
    # Split features into columns
    chunk_size = len(features) // 3 + 1
    
    for i, feature in enumerate(features):
        if i < chunk_size:
            with col1:
                inputs[feature] = st.number_input(feature, value=float(df[feature].mean()), format="%.4f")
        elif i < chunk_size * 2:
            with col2:
                inputs[feature] = st.number_input(feature, value=float(df[feature].mean()), format="%.4f")
        else:
            with col3:
                inputs[feature] = st.number_input(feature, value=float(df[feature].mean()), format="%.4f")
                
    # Initialize session state for prediction
    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False

    if st.button("Predecir Calidad", type="primary"):
        st.session_state.predict_clicked = True
        
    if st.session_state.predict_clicked:
        # --- Input Validation ---
        warnings = []
        for feature in features:
            min_val = df[feature].min()
            max_val = df[feature].max()
            user_val = inputs[feature]
            
            # Check if out of range
            if user_val < min_val or user_val > max_val:
                warnings.append(
                    f"⚠️ **{feature}**: {user_val:.2f} está fuera del rango observado "
                    f"({min_val:.2f} - {max_val:.2f})."
                )
        
        if warnings:
            st.warning("⚠️ **Advertencia: Valores fuera de rango**")
            st.caption("Los modelos fueron entrenados con vinos dentro de rangos específicos. Valores extremos pueden producir predicciones poco confiables.")
            for w in warnings:
                st.markdown(w)
            
            if not st.checkbox("Entiendo el problema, mostrar predicción de todos modos"):
                st.stop()
            
            st.markdown("---")
        # ------------------------

        # Prepare Input
        input_df = pd.DataFrame([inputs])
        
        # CRITICAL: Ensure columns are in the exact same order as training
        input_df = input_df[X.columns]
        
        input_scaled = scaler.transform(input_df)
        
        # Predictions
        pred_score = reg.predict(input_scaled)[0]
        pred_class = clf.predict(input_scaled)[0]
        pred_proba = clf.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.subheader("Resultados")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.metric("Puntaje Predicho (0-10)", f"{pred_score:.2f}")
            
            # Interpretación Granular de Calidad
            if pred_score < 5.0:
                st.error("Calidad: MALA (< 5.0)")
            elif 5.0 <= pred_score < 6.0:
                st.warning("Calidad: PROMEDIO / UMBRAL (5.0 - 6.0)")
            elif 6.0 <= pred_score < 7.0:
                st.success("Calidad: BUENA (6.0 - 7.0)")
            else:
                st.success("Calidad: EXCELENTE (> 7.0)")
                
        with c2:
            st.metric("Predicción de Clasificación", "Bueno" if pred_class == 1 else "Malo")
            st.write(f"**Confianza:** {max(pred_proba):.2%}")
            st.progress(max(pred_proba))
            
        # Reality Check
        st.markdown("### Comparativa de Referencia")
        st.markdown("Comparando su entrada con los valores promedio de **Vinos Buenos** (Calidad >= 6).")
        
        good_wines = df[df['calidad'] >= 6]
        
        # Calculate correlations dynamically
        correlations = df.corr()['calidad']
        
        # Compare ALL features
        with st.expander("Ver comparación detallada para todas las características", expanded=True):
            for feature in features:
                avg_good = good_wines[feature].mean()
                user_val = inputs[feature]
                corr = correlations[feature]
                
                # Determine direction based on correlation strength
                # Threshold of 0.1 for "meaningful" correlation
                if corr > 0.1:
                    direction = "positive" # Higher is better
                elif corr < -0.1:
                    direction = "negative" # Lower is better
                else:
                    direction = "neutral"
                
                col_rc1, col_rc2 = st.columns([1, 2])
                with col_rc1:
                    st.write(f"**{feature.title()}**")
                    st.caption(f"Corr: {corr:.2f}")
                
                with col_rc2:
                    if direction == "positive":
                        if user_val >= avg_good:
                            st.success(f"Mayor que el promedio ({avg_good:.2f}) - Positivo")
                        else:
                            st.warning(f"Menor que el promedio ({avg_good:.2f}) - Negativo")
                    elif direction == "negative":
                        if user_val <= avg_good:
                            st.success(f"Menor que el promedio ({avg_good:.2f}) - Positivo")
                        else:
                            st.warning(f"Mayor que el promedio ({avg_good:.2f}) - Negativo")
                    else:
                        # Neutral/Weak correlation
                        diff_pct = (user_val - avg_good) / avg_good * 100
                        if abs(diff_pct) < 10:
                            st.info(f"Cercano al promedio ({avg_good:.2f})")
                        elif user_val > avg_good:
                            st.write(f"Mayor que el promedio ({avg_good:.2f})")
                        else:
                            st.write(f"Menor que el promedio ({avg_good:.2f})")
