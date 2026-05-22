#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cuántos flujos de «Web Attack» hay en el CSV antiguo y en CIC-IDS2017 corregido."""
import glob, sys
import pandas as pd
viejo, corregido = sys.argv[1], sys.argv[2]
v = pd.read_csv(viejo, usecols=["Label"], encoding="utf-8")["Label"].str.strip().str.replace("�", "–")
print("CSV antiguo:")
print(v[v.str.startswith("Web Attack")].value_counts().to_string(), "\n  total:", v.str.startswith("Web Attack").sum())
c = pd.concat(pd.read_csv(f, usecols=["Label", "Attempted Category"]) for f in sorted(glob.glob(f"{corregido}/*.csv")))
real = c[(c["Attempted Category"] == -1) & c["Label"].str.startswith("Web Attack")]["Label"]
print("\nCIC-IDS2017 corregido (sin los «Attempted»):")
print(real.value_counts().to_string(), "\n  total:", len(real))
