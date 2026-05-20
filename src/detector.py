#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de intrusiones sobre CIC-IDS2017 CORREGIDO (DistriNet, KU Leuven).

Protocolo, fijado antes de ver ningún resultado:
  * Datos: los cinco CSV diarios de CICIDS2017_improved. Los flujos «Attempted» (Attempted Category != -1)
    pasan a BENIGN, como recomiendan los autores del dataset.
  * Fuera del modelo todo lo que identifica la conexión y no su comportamiento: id, Flow ID, IPs, puertos
    (origen y destino) y Timestamp. El modelo tiene que reconocer CÓMO se comporta un flujo, no a quién va.
  * División por TIEMPO: cada etiqueta, en orden cronológico, se parte en 60 % entrenamiento, 20 % validación
    y 20 % test (el tráfico benigno, igual pero por día). El modelo ve el principio de cada ataque y tiene que
    reconocer el final.
  * Dos tareas:
      1. DETECCIÓN: ataque o benigno. Métricas: tasa de detección, falsas alarmas por 10.000 flujos benignos, F1.
      2. FAMILIA: Benigno, DoS, DDoS, Fuerza bruta, Web, Escaneo, Botnet, Otros (Heartbleed, Infiltración).
  * Candidatos (con y sin pesos de clase) entrenados con el 60 % y elegidos SOLO con validación:
      detección -> mayor F1 de la clase «ataque»; familia -> mayor F1 macro.
  * El ganador se reentrena con el 80 % y se mide UNA vez en test.
Uso:  python src/detector.py --datos /data [--hilos 8] [--muestra 1.0]
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--datos", default="data"); ap.add_argument("--hilos", type=int, default=8)
ap.add_argument("--muestra", type=float, default=1.0, help="fracción para una prueba rápida (1.0 = todo)")
ap.add_argument("--salida", default="results")
a = ap.parse_args(); T0 = time.time()
def log(*x): print(f"[{time.time()-T0:5.0f}s]", *x, flush=True)

FAMILIA = {
    "BENIGN": "Benigno",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS Slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "FTP-Patator": "Fuerza bruta", "SSH-Patator": "Fuerza bruta",
    "Web Attack - Brute Force": "Web", "Web Attack - XSS": "Web", "Web Attack - SQL Injection": "Web",
    "Portscan": "Escaneo", "Infiltration - Portscan": "Escaneo",
    "Botnet": "Botnet",
    "Heartbleed": "Otros", "Infiltration": "Otros",
}
FUERA = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Attempted Category", "Label"]

# ---------- carga ----------
partes = []
for dia in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    d = pd.read_csv(os.path.join(a.datos, f"{dia}.csv"), low_memory=False)
    d["dia"] = dia
    partes.append(d)
df = pd.concat(partes, ignore_index=True)
if a.muestra < 1.0:
    df = df.sample(frac=a.muestra, random_state=42)
df["Label"] = df["Label"].str.strip()
df.loc[df["Attempted Category"] != -1, "Label"] = "BENIGN"
desconocidas = set(df["Label"]) - set(FAMILIA)
assert not desconocidas, f"etiquetas sin familia: {desconocidas}"
df["t"] = pd.to_datetime(df["Timestamp"], errors="coerce")
feats = [c for c in df.columns if c not in FUERA + ["dia", "t"]]
X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float32").replace([np.inf, -np.inf], np.nan)
ok = X.notna().all(axis=1) & df["t"].notna()
df, X = df[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
log(f"{len(df):,} flujos · {len(feats)} características · descartados por NaN/inf: {(~ok).sum():,}")

# ---------- división por tiempo ----------
parte = np.empty(len(df), dtype=object)
grupos = df.groupby(np.where(df["Label"] == "BENIGN", "BENIGN|" + df["dia"], df["Label"]))
for _, idx in grupos.groups.items():
    orden = df.loc[idx, "t"].sort_values(kind="stable").index.to_numpy()
    n = len(orden); c1, c2 = int(n * 0.6), int(n * 0.8)
    parte[orden[:c1]] = "train"; parte[orden[c1:c2]] = "val"; parte[orden[c2:]] = "test"
TR, VA, TE = parte == "train", parte == "val", parte == "test"
log(f"entrenamiento {TR.sum():,} | validación {VA.sum():,} | test {TE.sum():,}")

y_bin = (df["Label"] != "BENIGN").astype(int).to_numpy()
fams = sorted(set(FAMILIA.values()), key=lambda f: (f != "Benigno", f))
y_fam = df["Label"].map(FAMILIA).map({f: i for i, f in enumerate(fams)}).to_numpy()

def pesos(y, idx):
    cnt = np.bincount(y[idx]); w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    p = w[y[idx]]; return p / p.mean()

def entrenar(y, idx, ponderar, n_clases):
    kw = dict(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", random_state=42, n_jobs=a.hilos)
    if n_clases == 2: m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw)
    else: m = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **kw)
    clases = np.unique(y[idx]); mapa = np.full(int(y.max()) + 1, -1); mapa[clases] = np.arange(len(clases))
    m.fit(X.iloc[idx], mapa[y[idx]], sample_weight=pesos(y, idx) if ponderar else None, verbose=False)
    m.clases_reales = clases                     # para devolver las predicciones a las etiquetas originales
    return m

def det_metricas(yt, yp):
    ben = yt == 0
    return dict(f1_ataque=float(f1_score(yt, yp)), deteccion=float(recall_score(yt, yp)),
                precision=float(precision_score(yt, yp, zero_division=0)),
                falsas_alarmas_por_10k=float(yp[ben].mean() * 10000), exactitud=float(accuracy_score(yt, yp)))

itr, iva, ite, i80 = map(np.flatnonzero, (TR, VA, TE, TR | VA))
res = {"split": dict(train=len(itr), val=len(iva), test=len(ite)), "detector": {}, "familia": {}}

# ---------- 1. DETECCIÓN ----------
cands = {}
for nombre, pond in (("D0 binario", False), ("D1 binario con pesos", True)):
    m = entrenar(y_bin, itr, pond, 2); r = det_metricas(y_bin[iva], m.clases_reales[m.predict(X.iloc[iva])])
    cands[nombre] = (r, ("bin", pond)); log(f"VAL detección  {nombre:24} F1 {r['f1_ataque']:.4f} · detección {r['deteccion']:.4f} · FA/10k {r['falsas_alarmas_por_10k']:.1f}")
modelos_fam = {}   # el mismo modelo de familias sirve para D2/D3 y para F0/F1: se entrena una vez
for nombre, pond in (("D2 familia->ataque", False), ("D3 familia->ataque con pesos", True)):
    m = modelos_fam[pond] = entrenar(y_fam, itr, pond, len(fams)); r = det_metricas(y_bin[iva], (m.clases_reales[m.predict(X.iloc[iva])] != 0).astype(int))
    cands[nombre] = (r, ("fam", pond)); log(f"VAL detección  {nombre:24} F1 {r['f1_ataque']:.4f} · detección {r['deteccion']:.4f} · FA/10k {r['falsas_alarmas_por_10k']:.1f}")
res["detector"]["validacion"] = {k: v[0] for k, v in cands.items()}
mejor_d = max(cands, key=lambda k: cands[k][0]["f1_ataque"]); log(f"==> detector elegido por validación: {mejor_d}")

# ---------- 2. FAMILIA ----------
fc = {}
for nombre, pond in (("F0 familia", False), ("F1 familia con pesos", True)):
    m = modelos_fam[pond]
    f = f1_score(y_fam[iva], m.clases_reales[m.predict(X.iloc[iva])], labels=range(len(fams)), average="macro", zero_division=0)
    fc[nombre] = (float(f), pond); log(f"VAL familia    {nombre:24} F1 macro {f:.4f}")
res["familia"]["validacion"] = {k: v[0] for k, v in fc.items()}
mejor_f = max(fc, key=lambda k: fc[k][0]); log(f"==> clasificador de familia elegido por validación: {mejor_f}")

# ---------- 3. TEST, una sola vez ----------
tipo, pond = cands[mejor_d][1]
if tipo == "bin":
    md = entrenar(y_bin, i80, pond, 2); pd_te = md.clases_reales[md.predict(X.iloc[ite])]
else:
    md = entrenar(y_fam, i80, pond, len(fams)); pd_te = (md.clases_reales[md.predict(X.iloc[ite])] != 0).astype(int)
rd = det_metricas(y_bin[ite], pd_te); res["detector"]["elegido"] = mejor_d; res["detector"]["test"] = rd
log(f"TEST detección ({mejor_d}): F1 {rd['f1_ataque']:.4f} · detección {rd['deteccion']:.4f} · precisión {rd['precision']:.4f} · FA/10k {rd['falsas_alarmas_por_10k']:.1f}")
lab_te = df["Label"].to_numpy()[ite]
porlab = {l: dict(flujos=int((lab_te == l).sum()), detectados=float(pd_te[lab_te == l].mean())) for l in sorted(set(lab_te)) if l != "BENIGN"}
res["detector"]["test_por_ataque"] = porlab
for l, v in porlab.items(): log(f"   {l:28} {v['flujos']:>7,} flujos de test · detectado {v['detectados']:.2%}")

mf = entrenar(y_fam, i80, fc[mejor_f][1], len(fams)); pf = mf.clases_reales[mf.predict(X.iloc[ite])]
f_cl = f1_score(y_fam[ite], pf, labels=range(len(fams)), average=None, zero_division=0)
res["familia"]["elegido"] = mejor_f
res["familia"]["test"] = dict(f1_macro=float(f_cl.mean()), por_familia={fams[i]: dict(f1=float(f_cl[i]), soporte=int((y_fam[ite] == i).sum())) for i in range(len(fams))})
informe = classification_report(y_fam[ite], pf, labels=range(len(fams)), target_names=fams, digits=4, zero_division=0)
log(f"TEST familia ({mejor_f}): F1 macro {f_cl.mean():.4f}\n" + informe)

out = os.path.join(a.salida, "detector"); os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, "metricas.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(os.path.join(out, "informe_familias.txt"), "w", encoding="utf-8").write(informe)
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(confusion_matrix(y_fam[ite], pf, labels=range(len(fams))), display_labels=fams).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d")
ax.set_title("Familia del ataque · test"); plt.tight_layout(); plt.savefig(os.path.join(out, "matriz_familias.png"), dpi=120); plt.close()
imp = pd.Series(md.feature_importances_, index=feats).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 6)); imp.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Qué mira el detector"); plt.tight_layout(); plt.savefig(os.path.join(out, "importancias.png"), dpi=120); plt.close()
log("listo")
