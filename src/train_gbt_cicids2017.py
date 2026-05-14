#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento de un Gradient Boosted Tree (XGBoost) sobre CIC-IDS2017
para clasificar flujos de red en benignos o 14 tipos de ataque.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# -----------------------------------------------
# 1. CARGA Y PREPROCESADO DEL DATASET
# -----------------------------------------------

print("[*] Cargando el dataset CIC-IDS2017...")
df = pd.read_csv('cicids2017.csv', low_memory=False)

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()

print(f"    Shape inicial: {df.shape}")
print(f"    Distribución de clases:\n{df['Label'].value_counts()}\n")

# Reemplazar infinitos y eliminar NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print(f"    Shape después de limpieza: {df.shape}")

# Separar features y etiqueta
X = df.drop(columns=['Label'])
y_raw = df['Label']

# Codificar la etiqueta
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"    Clases encontradas: {list(le.classes_)}\n")

# -----------------------------------------------
# 2. DIVISIÓN TRAIN / TEST (80% / 20%)
# -----------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[*] Tamaño train: {X_train.shape[0]} muestras")
print(f"    Tamaño test:  {X_test.shape[0]} muestras\n")

# -----------------------------------------------
# 3. ENTRENAMIENTO DEL MODELO GBT
# -----------------------------------------------

print("[*] Entrenando el modelo XGBClassifier...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

print("[✓] Modelo entrenado correctamente!\n")

# -----------------------------------------------
# 4. EVALUACIÓN DEL MODELO
# -----------------------------------------------

print("[*] Evaluando el modelo sobre el conjunto de test...")
y_pred = model.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)
fig, ax = plt.subplots(figsize=(16, 14))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Matriz de Confusión - GBT sobre CIC-IDS2017', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("    [✓] Matriz de confusión guardada en confusion_matrix.png")

# Classification report
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("\n    Classification Report:\n")
print(report)

with open('classification_report.txt', 'w') as f:
    f.write(report)

# -----------------------------------------------
# 5. FEATURE IMPORTANCE
# -----------------------------------------------

print("[*] Generando gráfico de Feature Importance...")

importances = pd.Series(model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

top20 = importances_sorted.head(20)
bot10 = importances_sorted.tail(10)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

top20.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].invert_yaxis()
axes[0].set_title('Top 20 características más relevantes', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importancia (F-score)')

bot10.plot(kind='barh', ax=axes[1], color='salmon')
axes[1].invert_yaxis()
axes[1].set_title('Top 10 características menos relevantes', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importancia (F-score)')

plt.suptitle('Feature Importance - XGBClassifier (GBT) · CIC-IDS2017', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("    [✓] Gráfico guardado en feature_importance.png\n")

# -----------------------------------------------
# 6. CLASIFICACIÓN DE TRÁFICO REAL
# -----------------------------------------------

import os

csv_path = 'capture.pcap_Flow.csv'
if not os.path.exists(csv_path):
    csv_path = 'capture.csv'

if os.path.exists(csv_path):
    print(f"[*] Cargando fichero de tráfico real: {csv_path}")
    df_cap = pd.read_csv(csv_path, low_memory=False)
    df_cap.columns = df_cap.columns.str.strip()
    df_cap.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cap.fillna(0, inplace=True)

    # Alinear columnas con las del entrenamiento
    for col in set(X.columns) - set(df_cap.columns):
        df_cap[col] = 0
    df_cap = df_cap[X.columns]

    y_cap_pred = model.predict(df_cap)
    y_cap_labels = le.inverse_transform(y_cap_pred)

    print("\n    Predicciones del modelo sobre capture.pcap:")
    print(pd.Series(y_cap_labels).value_counts().to_string())

    df_cap['Predicted_Label'] = y_cap_labels
    df_cap.to_csv('capture_predictions.csv', index=False)
    print("\n    [✓] Predicciones guardadas en capture_predictions.csv\n")
else:
    print(f"[!] No se encontró el fichero {csv_path}.")
    print("    Hay que ejecutar CICFlowMeter sobre capture.pcap primero.\n")

print("[✓] Script completado.")
