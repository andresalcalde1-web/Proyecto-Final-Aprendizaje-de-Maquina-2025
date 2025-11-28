"""
Script para generar automáticamente la Tabla I del informe:
R² de Regresión Lineal y Polinomial para todas las variables
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Cargar datos
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

# Preparar datos
X = df.drop('calidad', axis=1)
y = df['calidad']

# Almacenar resultados
results = []

print("=" * 80)
print("CALCULANDO R² PARA TODAS LAS VARIABLES")
print("=" * 80)
print()

for feature in X.columns:
    print(f"Procesando: {feature}...", end=" ")
    
    # Datos de una sola variable
    X_single = X[[feature]]
    X_train, X_test, y_train, y_test = train_test_split(
        X_single, y, test_size=0.2, random_state=42
    )
    
    # 1. Regresión Lineal
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    r2_linear = r2_score(y_test, y_pred_lin)
    
    # 2. Regresión Polinomial (Grado 2)
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_test)
    r2_poly = r2_score(y_test, y_pred_poly)
    
    # Guardar resultados
    results.append({
        'Variable': feature,
        'R² Linear': r2_linear,
        'R² Polynomial (Deg 2)': r2_poly,
        'Improvement': r2_poly - r2_linear
    })
    
    print(f"✓ (Linear: {r2_linear:.4f}, Poly: {r2_poly:.4f})")

# Crear DataFrame y ordenar por R² Lineal descendente
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R² Linear', ascending=False)

print()
print("=" * 80)
print("TABLA I: REGRESSION PERFORMANCE (R²) FOR INDIVIDUAL FEATURES")
print("=" * 80)
print()
print(results_df.to_string(index=False))
print()

# Guardar a CSV para importar a LaTeX/Word
results_df.to_csv("regression_table.csv", index=False)
print("✓ Tabla guardada en: regression_table.csv")

# Generar formato Markdown para incluir directamente en informe
print()
print("=" * 80)
print("FORMATO MARKDOWN (para copiar al informe)")
print("=" * 80)
print()
print(results_df.to_markdown(index=False, floatfmt=".4f"))

# Estadísticas adicionales
print()
print("=" * 80)
print("ESTADÍSTICAS CLAVE")
print("=" * 80)
print(f"Mejor predictor (Linear): {results_df.iloc[0]['Variable']} (R²={results_df.iloc[0]['R² Linear']:.4f})")
print(f"Peor predictor (Linear): {results_df.iloc[-1]['Variable']} (R²={results_df.iloc[-1]['R² Linear']:.4f})")
print(f"Mayor mejora con Polynomial: {results_df.loc[results_df['Improvement'].idxmax()]['Variable']} (+{results_df['Improvement'].max():.4f})")
