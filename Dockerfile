FROM python:3.12-slim
# xgboost-cpu: la misma librería sin las dependencias de GPU (mucho más ligera)
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost-cpu matplotlib
RUN useradd -m analista
USER analista
WORKDIR /src
