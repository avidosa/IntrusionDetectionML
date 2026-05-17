#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Boosted Tree (XGBoost) sobre CIC-IDS2017, evaluado de cuatro maneras para ver cuánto de
la nota es fuga de datos y cuánto es aprendizaje de verdad.

  E0  tal cual: todas las columnas, división aleatoria 80/20
  E1  sin `Counter` (es el número de fila dentro de cada fichero diario, no una característica de red)
  E2  E1 + sin flujos duplicados (los duplicados caen a la vez en train y en test)
  E0c control: el modelo solo ve `Counter` (si acierta, `Counter` es un atajo y no una característica de red)
  E3  E2 + división por BLOQUES contiguos dentro de cada clase, en lugar de aleatoria
      (los flujos vecinos en el tiempo se parecen mucho; mezclados al azar, el test es casi una copia del train)

Uso:  python src/train_gbt_cicids2017.py E0 [--csv cicids2017.csv] [--hilos 8]
Salida: results/<experimento>/ (informe, matriz de confusión, importancias y metricas.json)
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

ap = argparse.ArgumentParser()
ap.add_argument("experimento", choices=["E0", "E0c", "E1", "E2", "E3"])
ap.add_argument("--csv", default="cicids2017.csv")
ap.add_argument("--hilos", type=int, default=8)
ap.add_argument("--salida", default="results")
a = ap.parse_args()
t0 = time.time()

# ---------- 1. carga y limpieza ----------
# el CSV trae el carácter de sustitución (U+FFFD) donde iba el guion largo de "Web Attack – Brute Force"
df = pd.read_csv(a.csv, low_memory=False, encoding="utf-8")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].str.strip().str.replace("\ufffd", "–", regex=False)
print(f"[*] {df.shape[0]:,} filas, {df.shape[1]} columnas")
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)   # el índice conserva el orden original
print(f"    tras quitar inf/NaN: {len(df):,}")

meta = {"experimento": a.experimento, "filas_tras_limpieza": int(len(df))}
if a.experimento not in ("E0", "E0c"):
    df = df.drop(columns=["Counter"])

if a.experimento in ("E2", "E3"):
    cols = [c for c in df.columns if c != "Label"]
    n0 = len(df)
    sin_dup = df.drop_duplicates(subset=cols + ["Label"], keep="first")
    # filas con EXACTAMENTE las mismas características pero otra etiqueta: ni el mejor modelo puede acertar ambas
    conflictos = len(sin_dup) - len(sin_dup.drop_duplicates(subset=cols, keep="first"))
    df = sin_dup
    meta.update(duplicados_eliminados=int(n0 - len(df)), filas_con_misma_huella_y_distinta_etiqueta=int(conflictos))
    print(f"    duplicados eliminados: {n0 - len(df):,} ({(n0 - len(df)) / n0:.1%}); "
          f"misma huella con otra etiqueta: {conflictos:,}")

X = (df[["Counter"]] if a.experimento == "E0c" else df.drop(columns=["Label"])).astype("float32")
le = LabelEncoder()
y = le.fit_transform(df["Label"])

# ---------- 2. división train / test ----------
if a.experimento == "E3":
    # cada clase se parte en 20 trozos contiguos (en el orden original); cada quinto trozo es test
    es_test = np.zeros(len(df), dtype=bool)
    for clase in np.unique(y):
        pos = np.flatnonzero(y == clase)                  # ya están en orden original
        trozo = max(1, int(np.ceil(len(pos) / 20)))
        es_test[pos[(np.arange(len(pos)) // trozo) % 5 == 4]] = True
    X_tr, X_te, y_tr, y_te = X[~es_test], X[es_test], y[~es_test], y[es_test]
else:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[*] train {len(X_tr):,} | test {len(X_te):,}")

# ---------- 3. modelo (los mismos hiperparámetros que la versión original) ----------
modelo = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=a.hilos)
t1 = time.time()
modelo.fit(X_tr, y_tr, verbose=False)
print(f"[✓] entrenado en {time.time() - t1:.0f} s con {a.hilos} hilos")

# ---------- 4. evaluación ----------
pred = modelo.predict(X_te)
etiquetas = np.arange(len(le.classes_))
informe = classification_report(y_te, pred, labels=etiquetas, target_names=le.classes_, digits=4, zero_division=0)
print(informe)
meta.update(
    exactitud=float(accuracy_score(y_te, pred)),
    f1_macro=float(f1_score(y_te, pred, average="macro", labels=etiquetas, zero_division=0)),
    f1_ponderado=float(f1_score(y_te, pred, average="weighted", labels=etiquetas, zero_division=0)),
    train=int(len(X_tr)), test=int(len(X_te)), segundos_entrenamiento=round(time.time() - t1),
    por_clase={c: {"f1": float(f), "soporte": int((y_te == i).sum())}
               for i, (c, f) in enumerate(zip(le.classes_, f1_score(y_te, pred, average=None, labels=etiquetas, zero_division=0)))},
)
imp = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
meta["top_importancias"] = {k: float(v) for k, v in imp.head(10).items()}

out = os.path.join(a.salida, a.experimento)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "informe.txt"), "w", encoding="utf-8").write(informe)
json.dump(meta, open(os.path.join(out, "metricas.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=etiquetas), display_labels=le.classes_).plot(
    ax=ax, cmap="Blues", xticks_rotation=60, colorbar=False, values_format="d")
ax.set_title(f"Matriz de confusión · {a.experimento}")
plt.tight_layout(); plt.savefig(os.path.join(out, "matriz_confusion.png"), dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(9, 7))
imp.head(15).iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
ax.set_title(f"Qué mira el modelo · {a.experimento}"); ax.set_xlabel("importancia")
plt.tight_layout(); plt.savefig(os.path.join(out, "importancias.png"), dpi=110); plt.close()
print(f"[✓] {a.experimento}: exactitud {meta['exactitud']:.4f} · F1 macro {meta['f1_macro']:.4f} · {time.time() - t0:.0f} s en total")
