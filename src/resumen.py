#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Junta los metricas.json de results/E0..E3 en una tabla (results/comparacion.md)."""
import json, os
filas, clases = [], {}
for e in ("E0", "E1", "E2", "E3"):
    p = os.path.join("results", e, "metricas.json")
    if os.path.exists(p):
        m = json.load(open(p, encoding="utf-8")); filas.append((e, m)); clases[e] = m["por_clase"]
out = ["| Experimento | Flujos de test | Exactitud | F1 macro |", "|---|---:|---:|---:|"]
for e, m in filas:
    out.append(f"| {e} | {m['test']:,} | {m['exactitud']:.4f} | {m['f1_macro']:.4f} |".replace(",", "."))
p = os.path.join("results", "E0c", "metricas.json")
if os.path.exists(p):
    c = json.load(open(p, encoding="utf-8"))
    out.append(f"| E0c (control: solo `Counter`) | {c['test']:,} | {c['exactitud']:.4f} | {c['f1_macro']:.4f} |".replace(",", "."))
out += ["", "F1 por clase:", "", "| Clase | " + " | ".join(e for e, _ in filas) + " | soporte (último) |",
        "|---|" + "---:|" * (len(filas) + 1)]
for c in clases[filas[0][0]]:
    out.append(f"| {c} | " + " | ".join(f"{clases[e][c]['f1']:.3f}" for e, _ in filas) + f" | {clases[filas[-1][0]][c]['soporte']} |")
open("results/comparacion.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
