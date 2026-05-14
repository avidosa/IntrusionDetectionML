# IntrusionDetectionML

Detector de ataques de red con aprendizaje supervisado: un clasificador **Gradient Boosted
Tree** entrenado sobre CIC-IDS2017 y aplicado después a una captura de tráfico real.

Mayo de 2026.

## Qué hace

1. Carga y limpia el dataset CIC-IDS2017 (flujos de red etiquetados, 78 características por flujo).
2. Entrena un `XGBClassifier` para distinguir 15 clases de tráfico: benigno, botnet, DDoS,
   cuatro variantes de DoS, fuerza bruta FTP y SSH, Heartbleed, infiltración, escaneo de puertos
   y tres ataques web (fuerza bruta, inyección SQL y XSS).
3. Evalúa con matriz de confusión, *classification report* e importancia de características.
4. Clasifica tráfico real: de un `.pcap` se extraen los flujos con **CICFlowMeter** y el modelo
   los etiqueta.

## Configuración del modelo

```python
XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8)
```

División 80/20 estratificada, `random_state=42`. Limpieza previa: normalización de nombres de
columna, infinitos a `NaN` y eliminación de filas incompletas.

## Resultados

Sobre 539.153 flujos de test: **exactitud 0,9999** y **media macro de F1 0,9918**.
El detalle por clase está en [`results/classification_report.txt`](results/classification_report.txt).

Las clases con menos margen son las esperables, las que tienen pocas muestras:
inyección SQL web (21 casos, F1 0,876), XSS (133 casos, F1 0,968) y fuerza bruta web
(311 casos, F1 0,985).

| Fichero | Qué contiene |
|---|---|
| `results/confusion_matrix.png` | matriz de confusión de las 15 clases |
| `results/feature_importance.png` | qué características pesan más en la decisión |
| `results/capture_predictions.csv` | predicciones sobre los flujos de la captura real |

## Limitaciones, que conviene leer

Las métricas son casi perfectas, y eso **no significa que el problema esté resuelto**.
CIC-IDS2017 es un dataset generado en laboratorio, con flujos duplicados y características
muy correlacionadas con cada ataque (el puerto de destino, por ejemplo, identifica casi por
sí solo algunas clases). Un número así se consigue con facilidad en este dataset y no se
traslada a tráfico de producción.

Una evaluación más honesta pediría: deduplicar los flujos, separar entrenamiento y test
**por tiempo** en lugar de al azar, y quitar las columnas que puedan funcionar como
identificador. Queda como siguiente paso, y es más interesante que el 0,9999.

## Reproducirlo

El dataset **no está en este repositorio** (son 783 MB). Se descarga de la Universidad de
New Brunswick (CIC-IDS2017, versión de flujos ya extraídos) y se deja como `cicids2017.csv`
junto al script:

```bash
pip install -r requirements.txt
python src/train_gbt_cicids2017.py
```
