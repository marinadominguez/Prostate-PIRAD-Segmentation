#!/usr/bin/env bash

# Edita el umbral según los resultados del Excel

# 1. Evaluar con el umbral seleccionado
python evaluation/testing_metrics_for_table_sigmoid.py --threshold 0.202

# 2. (Opcional) Calcular volúmenes
python evaluation/compute_prostate_volume.py --threshold 0.202
