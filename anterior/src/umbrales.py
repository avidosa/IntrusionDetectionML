#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajuste de umbrales por clase, SIN tocar el test para decidir nada.

Protocolo (fijado antes de ver resultados):
  * mismos datos y bloques que src/mejorar.py: 60 % entrenamiento, 20 % validación, 20 % test.
  * modelo C0 (el que eligió la validación) entrenado con el 60 %.
  * las 15 probabilidades se pasan a 13 sumando las tres variantes web.
  * a cada clase se le da un peso que multiplica su probabilidad antes del argmax (= mover su umbral).
    Se buscan por ascenso por coordenadas, maximizando el F1 macro (13 clases) en VALIDACIÓN.
  * solo se ajustan clases con >= 100 flujos en validación; las demás quedan con peso 1.
  * en TEST se mide UNA vez el mismo modelo con y sin pesos: la diferencia es solo el umbral.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="cicids2017.csv"); ap.add_argument("--hilos", type=int, default=8)
a = ap.parse_args(); T0 = time.time()
def log(*x): print(f"[{time.time()-T0:5.0f}s]", *x, flush=True)

df = pd.read_csv(a.csv, low_memory=False, encoding="utf-8")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].str.strip().str.replace("�", "–", regex=False)
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True).drop(columns=["Counter"])
cols = [c for c in df.columns if c != "Label"]
df = df.drop_duplicates(subset=cols + ["Label"], keep="first")
X = df[cols].astype("float32").reset_index(drop=True)
le = LabelEncoder(); y = le.fit_transform(df["Label"].reset_index(drop=True)); N = len(le.classes_)
web = [i for i, c in enumerate(le.classes_) if c.startswith("Web Attack")]

fold = np.zeros(len(y), dtype=int)
for c in range(N):
    pos = np.flatnonzero(y == c); trozo = max(1, int(np.ceil(len(pos) / 20)))
    fold[pos] = (np.arange(len(pos)) // trozo) % 5
TR = np.isin(fold, [0, 1, 3]); VA = fold == 2; TE = fold == 4

# espacio de 13 clases: columnas = clases no-web + una "Web Attack"
no_web = [i for i in range(N) if i not in web]
nombres13 = [le.classes_[i] for i in no_web] + ["Web Attack"]
def a13(v):
    m = {c: k for k, c in enumerate(no_web)}; w = len(no_web)
    return np.array([w if c in web else m[c] for c in v])
def prob13(P): return np.column_stack([P[:, no_web], P[:, web].sum(axis=1)])

m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=a.hilos)
m.fit(X[TR], y[TR], verbose=False); log("modelo C0 entrenado con el 60 %")
Pva, yva = prob13(m.predict_proba(X[VA])), a13(y[VA])
Pte, yte = prob13(m.predict_proba(X[TE])), a13(y[TE])
K = Pva.shape[1]; labs = np.arange(K)
f1m = lambda yt, P, w: f1_score(yt, np.argmax(P * w, axis=1), labels=labs, average="macro", zero_division=0)

w = np.ones(K)
ajustables = [k for k in range(K) if (yva == k).sum() >= 100]
log("clases ajustables:", [nombres13[k] for k in ajustables])
rejilla = np.exp(np.linspace(np.log(1/20), np.log(20), 41))
base_va = f1m(yva, Pva, w); log(f"VALIDACIÓN sin pesos: {base_va:.4f}")
for pasada in range(3):
    for k in ajustables:
        mejor, mejor_v = w[k], f1m(yva, Pva, w)
        for g in rejilla:
            w2 = w.copy(); w2[k] = g; v = f1m(yva, Pva, w2)
            if v > mejor_v + 1e-9: mejor, mejor_v = g, v
        w[k] = mejor
    log(f"pasada {pasada+1}: VALIDACIÓN con pesos {f1m(yva, Pva, w):.4f}")
log("pesos:", {nombres13[k]: round(float(w[k]), 3) for k in range(K) if abs(w[k]-1) > 1e-9})

# ---- TEST: una sola vez ----
sin = np.argmax(Pte, axis=1); con = np.argmax(Pte * w, axis=1)
r = {}
for nom, p in (("sin_pesos", sin), ("con_pesos", con)):
    f = f1_score(yte, p, labels=labs, average=None, zero_division=0)
    r[nom] = dict(f1_macro13=float(f.mean()), exactitud=float(accuracy_score(yte, p)),
                  por_clase={nombres13[k]: float(f[k]) for k in labs})
    log(f"TEST {nom:9}: F1 macro(13) {r[nom]['f1_macro13']:.4f} | exactitud {r[nom]['exactitud']:.4f}")
print(classification_report(yte, con, labels=labs, target_names=nombres13, digits=4, zero_division=0))
os.makedirs("results/umbrales", exist_ok=True)
open("results/umbrales/informe.txt", "w", encoding="utf-8").write(classification_report(yte, con, labels=labs, target_names=nombres13, digits=4, zero_division=0))
json.dump(dict(validacion_sin=base_va, validacion_con=f1m(yva, Pva, w), pesos={nombres13[k]: float(w[k]) for k in range(K)}, test=r),
          open("results/umbrales/metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
fig, ax = plt.subplots(figsize=(13, 11))
ConfusionMatrixDisplay(confusion_matrix(yte, con, labels=labs), display_labels=nombres13).plot(ax=ax, cmap="Blues", xticks_rotation=60, colorbar=False, values_format="d")
ax.set_title("Matriz de confusión · TEST · C0 con umbrales ajustados en validación")
plt.tight_layout(); plt.savefig("results/umbrales/matriz_confusion.png", dpi=110); plt.close()
log("listo")
