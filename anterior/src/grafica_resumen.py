#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gráfica de la versión anterior: cómo cae el F1 al quitar cada trampa, y el ajuste de umbrales."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.formatter.use_locale"] = False
from matplotlib.ticker import FuncFormatter
coma = FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").rstrip(".").replace(".", ","))
R = "anterior/results"
f1 = {e: json.load(open(f"{R}/{e}/metricas.json"))["f1_macro"] for e in ("E0", "E1", "E2", "E3")}
u = json.load(open(f"{R}/umbrales/metricas.json"))
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
nombres = ["E0  original", "E1  sin Counter", "E2  sin duplicados", "E3  división por bloques"]
vals = [f1[e] for e in ("E0", "E1", "E2", "E3")]
barras = a.barh(nombres[::-1], vals[::-1], color=["#4c72b0"] * 3 + ["#c44e52"])
for r, v in zip(barras, vals[::-1]): a.text(v + 0.003, r.get_y() + r.get_height() / 2, f"{v:.3f}".replace(".", ","), va="center")
a.set_xlim(0.80, 1.02); a.xaxis.set_major_formatter(coma); a.set_xlabel("F1 macro (15 clases)")
a.set_title("Quitando trampas, una a una")
x = [0, 1]
a2 = [u["validacion_sin"], u["validacion_con"]]; t2 = [u["test"]["sin_pesos"]["f1_macro13"], u["test"]["con_pesos"]["f1_macro13"]]
b.plot(x, a2, "o-", label="validación", color="#4c72b0", lw=2)
b.plot(x, t2, "o-", label="prueba", color="#c44e52", lw=2)
for xs, ys in ((x, a2), (x, t2)):
    for xi, yi in zip(xs, ys): b.text(xi, yi + 0.003, f"{yi:.3f}".replace(".", ","), ha="center")
b.set_xticks(x, ["sin ajustar", "umbrales ajustados"]); b.set_xlim(-0.4, 1.4); b.set_ylim(0.88, 0.98); b.yaxis.set_major_formatter(coma)
b.set_ylabel("F1 macro (13 clases)"); b.set_title("Ajustar umbrales:\nsube en validación, baja en prueba")
b.legend(loc="lower left")
plt.tight_layout(); plt.savefig(f"{R}/resumen.png", dpi=120); plt.close()
print("ok")
