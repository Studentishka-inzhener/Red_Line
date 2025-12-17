#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py

Скрипт для генерации и сохранения примерных данных физико-математических моделей
авиационных и ракетных двигателей для обучения нейросети engine_analyzer.py.

Выход:
  data/
    aviation/model_000.csv ...
    rocket/model_000.csv ...
  labels.csv  (для supervised режимов)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Конфигурация
# -----------------------------
N_AVIATION = 50    # количество авиационных моделей
N_ROCKET = 50      # количество ракетных моделей
T = 1000           # число временных точек
PARAMS_AV = ["temperature", "pressure", "rpm", "fuel_flow", "thrust"]
PARAMS_RK = ["chamber_temp", "chamber_pressure", "mass_flow", "nozzle_eff", "thrust"]

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

# -----------------------------
# Функции генерации
# -----------------------------
def generate_aviation_model():
    """
    Синтетическая модель авиационного двигателя:
      - нормальные режимы с шумом
      - редкие аномалии (вибрации, скачки температуры)
    """
    t = np.linspace(0, 10, T)
    temp = 600 + 50*np.sin(0.2*t) + np.random.normal(0, 5, T)
    press = 1.5 + 0.05*np.cos(0.3*t) + np.random.normal(0, 0.02, T)
    rpm = 9000 + 200*np.sin(0.5*t) + np.random.normal(0, 50, T)
    fuel = 0.8 + 0.02*np.sin(0.4*t) + np.random.normal(0, 0.01, T)
    thrust = 20 + 2*np.sin(0.25*t) + np.random.normal(0, 0.2, T)
    data = np.vstack([temp, press, rpm, fuel, thrust]).T

    # случайная аномалия
    if np.random.rand() < 0.2:
        idx = np.random.randint(100, T-100)
        data[idx:idx+50, 0] += np.random.uniform(100, 200)  # резкий перегрев
    return data


def generate_rocket_model():
    """
    Синтетическая модель ракетного двигателя:
      - пульсации давления и температуры
      - редкие нарушения подачи топлива
    """
    t = np.linspace(0, 5, T)
    chamber_temp = 2500 + 100*np.sin(0.8*t) + np.random.normal(0, 30, T)
    chamber_press = 20 + 2*np.cos(0.6*t) + np.random.normal(0, 0.3, T)
    mass_flow = 50 + 5*np.sin(0.7*t) + np.random.normal(0, 1.0, T)
    nozzle_eff = 0.95 + 0.02*np.sin(0.4*t) + np.random.normal(0, 0.005, T)
    thrust = 500 + 50*np.sin(0.5*t) + np.random.normal(0, 5, T)
    data = np.vstack([chamber_temp, chamber_press, mass_flow, nozzle_eff, thrust]).T

    # редкая аномалия: падение давления и тяги
    if np.random.rand() < 0.2:
        idx = np.random.randint(200, T-200)
        data[idx:idx+100, 1] -= np.random.uniform(3, 5)
        data[idx:idx+100, 4] -= np.random.uniform(50, 100)
    return data

# -----------------------------
# Генерация данных
# -----------------------------
def save_models(engine_type, n_models, param_names, generator_func):
    out_dir = os.path.join(OUT_DIR, engine_type)
    os.makedirs(out_dir, exist_ok=True)
    all_paths = []
    for i in tqdm(range(n_models), desc=f"Generating {engine_type} models"):
        data = generator_func()
        df = pd.DataFrame(data, columns=param_names)
        path = os.path.join(out_dir, f"model_{i:03d}.csv")
        df.to_csv(path, index=False)
        all_paths.append(path)
    return all_paths

def create_labels(aviation_paths, rocket_paths, out_csv="labels.csv"):
    """
    Создаёт файл с метками (для supervised обучения).
    Метка = эффективность двигателя (чем ближе к номиналу, тем выше).
    Аномалии получают меньший score.
    """
    labels = []
    # авиация
    for p in aviation_paths:
        df = pd.read_csv(p)
        # условная метка: средняя тяга / температура
        score = df["thrust"].mean() / df["temperature"].mean() * 10
        # penalty for anomalies
        if df["temperature"].max() > 800:
            score *= 0.8
        labels.append({"path": p, "label": score})
    # ракета
    for p in rocket_paths:
        df = pd.read_csv(p)
        score = df["thrust"].mean() / df["chamber_temp"].mean() * 100
        if df["chamber_pressure"].min() < 17:
            score *= 0.7
        labels.append({"path": p, "label": score})

    df_labels = pd.DataFrame(labels)
    df_labels.to_csv(out_csv, index=False)
    print(f"Saved labels to {out_csv} (total {len(df_labels)} records)")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    av_paths = save_models("aviation", N_AVIATION, PARAMS_AV, generate_aviation_model)
    rk_paths = save_models("rocket", N_ROCKET, PARAMS_RK, generate_rocket_model)
    create_labels(av_paths, rk_paths, out_csv=os.path.join(OUT_DIR, "labels.csv"))
    print("\n Датасет приготовлен")
    print("Папки: data/aviation/, data/rocket/, метки: data/labels.csv")
    print("\nПример запуска обучения:")
    print("python engine_analyzer.py --files data/aviation/*.csv data/rocket/*.csv --labels data/labels.csv --task regression --engine aviation")
