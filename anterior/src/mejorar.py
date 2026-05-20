#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mejorar el detector SIN hacer trampa.

Protocolo (declarado antes de mirar ningún resultado):
  * mismos datos que E3: sin `Counter`, sin duplicados, y partidos en BLOQUES contiguos por clase.
  * cada clase se parte en 20 bloques; fold = bloque % 5  ->  4: TEST (20 %), 2: VALIDACIÓN (20 %), resto: ENTRENAMIENTO (60 %)
  * cada mejora se prueba entrenando con el 60 % y midiendo en VALIDACIÓN. El test no se toca.
  * criterio de elección: mayor F1 macro de validación en el espacio de 13 clases (con las tres variantes
    web unidas), para poder comparar modelos que entrenan con 15 y con 13 clases.
  * la mejor configuración se reentrena con entrenamiento + validación (80 %) y se mide UNA vez en TEST.

Mejoras que se prueban: pesar las clases raras, un modelo más grande, submuestrear el tráfico benigno y
unir XSS, fuerza bruta e inyección SQL web en una sola clase (a nivel de flujo son casi idénticas).
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="cicids2017.csv")
ap.add_argument("--hilos", type=int, default=8)
a = ap.parse_args()
T0 = time.time()
def log(*x): print(f"[{time.time()-T0:5.0f}s]", *x, flush=True)

# ---------- datos (idénticos a E3) ----------
df = pd.read_csv(a.csv, low_memory=False, encoding="utf-8")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].str.strip().str.replace("�", "–", regex=False)
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True).drop(columns=["Counter"])
cols = [c for c in df.columns if c != "Label"]
df = df.drop_duplicates(subset=cols + ["Label"], keep="first")
X = df[cols].astype("float32").reset_index(drop=True)
etiquetas_txt = df["Label"].reset_index(drop=True)
le = LabelEncoder(); y = le.fit_transform(etiquetas_txt)
N = len(le.classes_)
web = [i for i, c in enumerate(le.classes_) if c.startswith("Web Attack")]
log(f"{len(X):,} flujos, {N} clases; web = {[le.classes_[i] for i in web]}")

# ---------- bloques ----------
fold = np.zeros(len(y), dtype=int)
for c in range(N):
    pos = np.flatnonzero(y == c)
    trozo = max(1, int(np.ceil(len(pos) / 20)))
    fold[pos] = (np.arange(len(pos)) // trozo) % 5
TR = np.isin(fold, [0, 1, 3]); VA = fold == 2; TE = fold == 4
log(f"entrenamiento {TR.sum():,} | validación {VA.sum():,} | test {TE.sum():,}")

# espacio de 13 clases: las tres variantes web -> una
def a13(v):
    v = v.copy(); v[np.isin(v, web)] = web[0]; return v
y13 = a13(y)

def metricas(yt, yp, es13):
    """F1 macro en el espacio de 13 clases (comparable entre modelos); yt/yp ya en ese espacio."""
    labs = np.unique(y13)
    f = f1_score(yt, yp, labels=labs, average=None, zero_division=0)
    sop = np.array([(yt == l).sum() for l in labs])
    return dict(f1_macro13=float(f.mean()), f1_macro_grandes=float(f[sop >= 100].mean()),
                exactitud=float(accuracy_score(yt, yp)))

# ---------- configuraciones a probar ----------
CONFIG = {
    "C0 base":                   dict(pesos=False, arboles=200, prof=6, benign=None, unir=False),
    "C1 + pesos de clase":       dict(pesos=True,  arboles=200, prof=6, benign=None, unir=False),
    "C2 + modelo mas grande":    dict(pesos=True,  arboles=400, prof=8, benign=None, unir=False),
    "C3 + submuestreo benigno":  dict(pesos=True,  arboles=200, prof=6, benign=300_000, unir=False),
    "C4 + web unido":            dict(pesos=True,  arboles=200, prof=6, benign=300_000, unir=True),
    "C5 web unido y grande":     dict(pesos=True,  arboles=400, prof=8, benign=300_000, unir=True),
}

def entrenar(cfg, idx_train):
    rng = np.random.default_rng(42)
    idx = idx_train
    yy = y13 if cfg["unir"] else y
    if cfg["benign"]:
        b = np.flatnonzero(etiquetas_txt.values == "Benign")
        quitar = np.setdiff1d(b[np.isin(b, idx)], rng.choice(b[np.isin(b, idx)], min(cfg["benign"], np.isin(b, idx).sum()), replace=False))
        idx = np.setdiff1d(idx, quitar)
    peso = None
    if cfg["pesos"]:
        cnt = pd.Series(yy[idx]).value_counts()
        w = 1.0 / np.sqrt(cnt)
        peso = pd.Series(yy[idx]).map(w).values; peso = peso / peso.mean()
    clases = np.unique(yy[idx])
    remap = {c: i for i, c in enumerate(clases)}
    m = XGBClassifier(n_estimators=cfg["arboles"], max_depth=cfg["prof"], learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=a.hilos)
    m.fit(X.iloc[idx], np.vectorize(remap.get)(yy[idx]), sample_weight=peso, verbose=False)
    def predecir(Xp):
        return clases[m.predict(Xp)]
    return m, predecir

# ---------- 1) elegir con VALIDACIÓN ----------
idx_tr = np.flatnonzero(TR); idx_va = np.flatnonzero(VA)
resultados = {}
for nombre, cfg in CONFIG.items():
    t = time.time()
    _, pred = entrenar(cfg, idx_tr)
    yp = a13(pred(X.iloc[idx_va]))
    r = metricas(y13[idx_va], yp, True); r["segundos"] = round(time.time() - t)
    resultados[nombre] = r
    log(f"{nombre:28} VALIDACIÓN  F1 macro(13) {r['f1_macro13']:.4f} | clases grandes {r['f1_macro_grandes']:.4f} | exactitud {r['exactitud']:.4f} | {r['segundos']} s")

mejor = max(resultados, key=lambda k: resultados[k]["f1_macro13"])
log(f"==> elegida por validación: {mejor}")

# ---------- 2) reentrenar con 80 % y medir UNA vez en TEST ----------
cfg = CONFIG[mejor]
idx_80 = np.flatnonzero(TR | VA); idx_te = np.flatnonzero(TE)
t = time.time()
modelo, pred = entrenar(cfg, idx_80)
yp = a13(pred(X.iloc[idx_te])); yt = y13[idx_te]
final = metricas(yt, yp, True)
log(f"TEST (una sola vez) con «{mejor}»: F1 macro(13) {final['f1_macro13']:.4f} | clases grandes {final['f1_macro_grandes']:.4f} | exactitud {final['exactitud']:.4f} | {time.time()-t:.0f} s")

# referencia: el modelo base C0 reentrenado igual y medido igual en el MISMO test
_, pred0 = entrenar(CONFIG["C0 base"], idx_80)
base = metricas(yt, a13(pred0(X.iloc[idx_te])), True)
log(f"TEST referencia C0 base (mismo test): F1 macro(13) {base['f1_macro13']:.4f} | clases grandes {base['f1_macro_grandes']:.4f} | exactitud {base['exactitud']:.4f}")

labs = np.unique(y13)
nombres13 = [("Web Attack" if l == web[0] else le.classes_[l]) for l in labs]
informe = classification_report(yt, yp, labels=labs, target_names=nombres13, digits=4, zero_division=0)
print(informe)
os.makedirs("results/mejora", exist_ok=True)
open("results/mejora/informe.txt", "w", encoding="utf-8").write(informe)
json.dump(dict(eleccion=mejor, validacion=resultados, test_final=final, test_base=base,
               split=dict(entrenamiento=int(TR.sum()), validacion=int(VA.sum()), test=int(TE.sum()))),
          open("results/mejora/metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
fig, ax = plt.subplots(figsize=(13, 11))
ConfusionMatrixDisplay(confusion_matrix(yt, yp, labels=labs), display_labels=nombres13).plot(
    ax=ax, cmap="Blues", xticks_rotation=60, colorbar=False, values_format="d")
ax.set_title(f"Matriz de confusión · TEST · {mejor}")
plt.tight_layout(); plt.savefig("results/mejora/matriz_confusion.png", dpi=110); plt.close()
log("listo")
