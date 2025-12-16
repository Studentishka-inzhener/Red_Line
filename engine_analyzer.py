#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine_analyzer.py

Полносвязная нейросеть (MLP) для анализа физико-математических моделей авиационных и ракетных двигателей.
Поддерживает supervised (regression/classification) и unsupervised (autoencoder для обнаружения аномалий).
Консольный интерфейс: выбор типа двигателя и добавление файлов модели (CSV/JSON/NPZ/MAT/TXT).

Автор: Цибрев-Аруев Д.А.
Дата: 30.10.2025
"""

import os
import sys
import argparse
import json
import math
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utilities for file parsing
# -----------------------------
def load_file_generic(path: str) -> dict:
    """
    Попытаться загрузить файл с данными модели. Возвращает словарь:
    {
      'data': np.ndarray (2D: samples x channels/time or 1D vector),
      'meta': dict (опционально)
    }
    Поддерживаемые форматы: .csv, .json, .npz, .npy, .mat, .txt
    """
    ext = os.path.splitext(path)[1].lower()
    result = {'data': None, 'meta': {}}
    if ext == '.csv':
        df = pd.read_csv(path)
        result['data'] = df.values.astype(float)
        result['meta'] = {'columns': list(df.columns)}
    elif ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            js = json.load(f)
        # Ожидаем либо {"data": [...]} либо табличный список объектов
        if isinstance(js, dict) and 'data' in js:
            arr = np.array(js['data'], dtype=float)
            result['data'] = arr
            result['meta'] = {k: v for k, v in js.items() if k != 'data'}
        elif isinstance(js, list):
            df = pd.json_normalize(js)
            result['data'] = df.values.astype(float)
            result['meta'] = {'columns': list(df.columns)}
        else:
            raise ValueError("JSON формат не поддерживается: ожидается массив или ключ 'data'")
    elif ext in ('.npz', '.npy'):
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # брать первый массив
            k = list(arr.files)[0]
            result['data'] = arr[k].astype(float)
        else:
            result['data'] = arr.astype(float)
    elif ext == '.mat':
        mat = loadmat(path)
        # взять первое подходящее поле с числовыми данными
        for k, v in mat.items():
            if k.startswith('__'):
                continue
            if isinstance(v, np.ndarray):
                result['data'] = v.astype(float)
                result['meta'] = {'mat_key': k}
                break
        if result['data'] is None:
            raise ValueError("Не найден пригодный массив в .mat файле")
    elif ext in ('.txt', '.dat'):
        arr = np.loadtxt(path)
        result['data'] = arr.astype(float)
    else:
        raise ValueError(f"Неизвестный формат файла: {ext}")
    # Ensure 2D
    if result['data'] is not None:
        arr = np.array(result['data'])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        result['data'] = arr
    return result

# -----------------------------
# Feature extraction
# -----------------------------
def extract_time_series_features(arr: np.ndarray, sr: Optional[float] = None) -> np.ndarray:
    """
    arr: 2D array (T x channels) - временной ряд или 1D vector shaped as (T,1)
    Возвращает 1D вектор признаков.
    Извлекаются:
      - статистика по каждому каналу: mean, std, median, min, max, skew, kurtosis (approx)
      - производные (first, second) статистики
      - энергия в нескольких FFT-диапазонах
      - peak-to-peak, rms
    """
    import scipy.stats as stats
    T, C = arr.shape
    feats = []
    for ch in range(C):
        x = arr[:, ch].astype(float)
        # basic stats
        mean = x.mean()
        std = x.std(ddof=0)
        med = np.median(x)
        mn = x.min()
        mx = x.max()
        ptp = mx - mn
        rms = math.sqrt(np.mean(x**2))
        # robust measures
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        # skew/kurtosis (use scipy)
        try:
            skew = float(stats.skew(x))
            kurt = float(stats.kurtosis(x))
        except Exception:
            skew, kurt = 0.0, 0.0
        feats += [mean, std, med, mn, mx, ptp, rms, q25, q75, iqr, skew, kurt]
        # derivatives
        dx = np.gradient(x)
        ddx = np.gradient(dx)
        feats += [dx.mean(), dx.std(), ddx.mean(), ddx.std()]

        # FFT energy bands
        n = len(x)
        if n >= 8:
            yf = np.abs(rfft(x - mean))
            freqs = rfftfreq(n, d=1.0 if sr is None else 1.0/sr)
            # define bands relative to Nyquist: split into 4 bands
            nb = 4
            L = len(yf)
            band_size = max(1, L // nb)
            for b in range(nb):
                s = b * band_size
                e = min(L, (b + 1) * band_size)
                band_energy = yf[s:e].sum()
                feats.append(band_energy)
            feats.append(yf.sum())  # total energy
        else:
            feats += [0.0] * 5
    # flatten to numpy
    return np.array(feats, dtype=float)

def extract_features_from_sample(sample: np.ndarray) -> np.ndarray:
    """
    sample: 2D or 1D ndarray representing a single model output / time series / parameter set
    If sample has many rows (time), treat as time-series; else treat each column as feature.
    """
    arr = np.array(sample)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # If time dimension large (>10), consider time-series features
    if arr.shape[0] >= 10:
        return extract_time_series_features(arr)
    else:
        # If small, compute simple stats across columns
        v = arr.flatten()
        feats = [
            v.mean(),
            v.std(ddof=0) if v.size > 1 else 0.0,
            v.min(),
            v.max(),
            np.median(v),
            np.percentile(v, 25) if v.size > 1 else v[0],
            np.percentile(v, 75) if v.size > 1 else v[0],
        ]
        return np.array(feats, dtype=float)

# -----------------------------
# Dataset
# -----------------------------
class EngineDataset(Dataset):
    """
    Хранилище признаков и опциональных меток.
    inputs: ndarray (N x F)
    labels: ndarray (N x ...) or None
    meta: list of dicts
    """
    def __init__(self, inputs: np.ndarray, labels: Optional[np.ndarray] = None, meta: Optional[List[dict]] = None):
        self.X = inputs.astype(np.float32)
        self.y = labels.astype(np.float32) if labels is not None else None
        self.meta = meta if meta is not None else [None] * len(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx], self.meta[idx]
        else:
            return self.X[idx], self.y[idx], self.meta[idx]

# -----------------------------
# Models: MLP and Autoencoder
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        act = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}.get(activation, nn.ReLU)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int], activation: str = 'relu'):
        super().__init__()
        act = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}.get(activation, nn.ReLU)
        # encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h)); enc_layers.append(act()); prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        # decoder (mirror)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h)); dec_layers.append(act()); prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec

# -----------------------------
# Training & evaluation helpers
# -----------------------------
def train_supervised(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     device: torch.device,
                     epochs: int = 100,
                     lr: float = 1e-3,
                     weight_decay: float = 0.0,
                     task: str = 'regression',
                     early_stop_patience: int = 10,
                     checkpoint_path: str = 'best_supervised.pth'):
    """
    task: 'regression' or 'classification'
    """
    model.to(device)
    if task == 'regression':
        criterion = nn.MSELoss()
    else:
        # For classification binary or multi-class: use BCE or CrossEntropy based on labels shape
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val = float('inf')
    best_epoch = -1
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            if task == 'regression':
                x, y, *_ = batch
                y = y.to(device)
            else:
                x, y, *_ = batch
                y = y.long().to(device)
            x = x.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y) if task == 'regression' else criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if task == 'regression':
                    x, y, *_ = batch
                    y = y.to(device)
                else:
                    x, y, *_ = batch
                    y = y.long().to(device)
                x = x.to(device)
                pred = model(x)
                loss = criterion(pred.squeeze(), y) if task == 'regression' else criterion(pred, y)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # early stopping & checkpoint
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, checkpoint_path)
        if epoch - best_epoch >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # load best
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
    return model, history

def train_autoencoder(ae: Autoencoder,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      device: torch.device,
                      epochs: int = 100,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      early_stop_patience: int = 10,
                      checkpoint_path: str = 'best_autoencoder.pth'):
    ae.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val = float('inf'); best_epoch = -1
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs + 1):
        ae.train()
        t_losses = []
        for batch in train_loader:
            x, *_ = batch
            x = x.to(device)
            optimizer.zero_grad()
            xrec = ae(x)
            loss = criterion(xrec, x)
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())
        train_loss = float(np.mean(t_losses)) if t_losses else 0.0

        # validation
        ae.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, *_ = batch
                x = x.to(device)
                xrec = ae(x)
                loss = criterion(xrec, x)
                v_losses.append(loss.item())
        val_loss = float(np.mean(v_losses)) if v_losses else 0.0
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"[AE Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val - 1e-9:
            best_val = val_loss; best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state': ae.state_dict(), 'optimizer_state': optimizer.state_dict()}, checkpoint_path)
        if epoch - best_epoch >= early_stop_patience:
            print("Early stopping AE.")
            break
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        ae.load_state_dict(ckpt['model_state'])
    return ae, history

# -----------------------------
# Evaluation & plotting
# -----------------------------
def evaluate_supervised(model: nn.Module, loader: DataLoader, device: torch.device, task: str = 'regression'):
    model.eval()
    preds = []
    targets = []
    metas = []
    with torch.no_grad():
        for batch in loader:
            if task == 'regression':
                x, y, meta = batch
                targets.append(y.numpy())
            else:
                x, y, meta = batch
                targets.append(y.numpy())
            x = x.to(device)
            out = model(x).cpu().numpy()
            preds.append(out)
            metas += list(meta)
    preds = np.vstack(preds)
    targets = np.vstack(targets).squeeze()
    return preds, targets, metas

def evaluate_autoencoder(ae: Autoencoder, loader: DataLoader, device: torch.device):
    ae.eval()
    rec_errors = []
    metas = []
    with torch.no_grad():
        for batch in loader:
            x, meta = batch
            x = x.to(device)
            xrec = ae(x).cpu().numpy()
            x_np = x.cpu().numpy()
            # MSE per sample
            mse = np.mean((xrec - x_np)**2, axis=1)
            rec_errors.append(mse)
            metas += list(meta)
    rec_errors = np.concatenate(rec_errors)
    return rec_errors, metas

def plot_training_history(history: dict, out_dir: str, title: str = 'training'):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Loss curve ({title})')
    plt.legend()
    path = os.path.join(out_dir, f'loss_{title}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved loss plot to", path)

def save_results_table(df: pd.DataFrame, out_dir: str, name: str = 'results'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{name}.csv')
    df.to_csv(path, index=False)
    print("Saved results table to", path)

# -----------------------------
# High-level pipeline
# -----------------------------
def build_feature_matrix_from_files(file_paths: List[str]) -> Tuple[np.ndarray, List[dict]]:
    """
    Для каждого файла пытаемся извлечь признаки.
    Возвращаем: features (N x F), meta_list
    """
    feats = []
    meta = []
    for p in file_paths:
        try:
            loaded = load_file_generic(p)
            arr = loaded['data']
            f = extract_features_from_sample(arr)
            feats.append(f)
            m = {'path': p}
            m.update(loaded.get('meta', {}))
            meta.append(m)
        except Exception as e:
            print(f"Warning: не удалось обработать {p}: {e}")
    X = np.vstack(feats).astype(float)
    return X, meta

def prepare_datasets(X: np.ndarray, y: Optional[np.ndarray], test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    if y is None:
        # unsupervised: split train/val/test only X
        X_train, X_temp = train_test_split(X, test_size=test_size+val_size, random_state=random_state)
        val_prop = val_size / (test_size + val_size)
        X_val, X_test = train_test_split(X_temp, test_size=val_prop, random_state=random_state)
        return X_train, X_val, X_test, None, None, None
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, random_state=random_state)
        val_prop = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_prop, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

# -----------------------------
# Console / CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Engine model analyzer (MLP / Autoencoder)")
    p.add_argument('--engine', choices=['aviation', 'rocket'], required=False, help='Тип двигателя (авиационный/ракетный)')
    p.add_argument('--files', nargs='+', help='Файлы с моделями (CSV/JSON/NPZ/MAT/TXT)')
    p.add_argument('--labels', help='CSV файл с метками (если есть). Должен содержать колонку path и колонку label (float для регрессии или int для классификации)')
    p.add_argument('--task', choices=['regression', 'classification', 'anomaly'], default='anomaly', help='Режим задачи')
    p.add_argument('--out', default='out_results', help='Папка для результатов')
    p.add_argument('--epochs', type=int, default=100, help='Эпохи обучения')
    p.add_argument('--batch', type=int, default=32, help='batch size')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='cuda or cpu')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Device:", args.device)
    device = torch.device(args.device)

    # Interactive fallback if not provided
    if args.engine is None:
        print("Выберите тип двигателя:\n 1) aviation\n 2) rocket")
        ch = input("Введите 1 или 2: ").strip()
        args.engine = 'aviation' if ch == '1' else 'rocket'

    if not args.files:
        print("Укажите пути к файлам модели (через пробел). Пример: model1.csv model2.mat")
        files_input = input("files: ").strip()
        files = files_input.split()
    else:
        files = args.files

    # Build features
    print(f"Обрабатываем {len(files)} файлов...")
    X_raw, metas = build_feature_matrix_from_files(files)
    print("Форма матрицы признаков:", X_raw.shape)

    # If labels provided, load them
    y = None
    if args.labels:
        df_lbl = pd.read_csv(args.labels)
        # expect columns "path" and "label"
        label_map = {row['path']: row['label'] for _, row in df_lbl.iterrows()}
        lbls = []
        for m in metas:
            p = os.path.basename(m['path'])
            # try full path then basename
            lbl = label_map.get(m['path'], label_map.get(p, None))
            if lbl is None:
                raise ValueError(f"Отсутствует метка для файла {m['path']} в {args.labels}")
            lbls.append(lbl)
        y = np.array(lbls)
        print("Загружены метки, задача:", args.task)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Prepare datasets
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(X_scaled, y, test_size=0.2, val_size=0.1)
    out_dir = os.path.join(args.out, f"{args.engine}_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    if args.task == 'anomaly':
        # Autoencoder pipeline
        input_dim = X_train.shape[1]
        ae = Autoencoder(input_dim=input_dim, latent_dim=max(4, input_dim // 4), hidden_dims=[max(32, input_dim//2)], activation='relu')
        train_ds = EngineDataset(X_train)
        val_ds = EngineDataset(X_val)
        test_ds = EngineDataset(X_test)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)
        print("Training autoencoder...")
        ae, history = train_autoencoder(ae, train_loader, val_loader, device, epochs=args.epochs, lr=1e-3, early_stop_patience=10, checkpoint_path=os.path.join(out_dir, 'ae_best.pth'))
        plot_training_history(history, out_dir, title='autoencoder')
        # compute reconstruction errors
        rec_train, _ = evaluate_autoencoder(ae, train_loader, device)
        rec_val, metas_val = evaluate_autoencoder(ae, val_loader, device)
        rec_test, metas_test = evaluate_autoencoder(ae, test_loader, device)
        # threshold (e.g. mean + 3*std on validation)
        thr = rec_val.mean() + 3 * rec_val.std()
        print(f"Threshold for anomaly (val mean+3std): {thr:.6f}")
        # report
        df_res = pd.DataFrame({
            'path': [m['path'] for m in metas_test],
            'reconstruction_error': rec_test,
            'anomaly': rec_test > thr
        })
        save_results_table(df_res, out_dir, name='anomaly_results')
        # plot histogram
        plt.figure()
        plt.hist(rec_test, bins=50)
        plt.axvline(thr, color='r', linestyle='--', label='threshold')
        plt.xlabel('reconstruction error')
        plt.ylabel('count')
        plt.title('Anomaly scores (test)')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'anomaly_hist.png'), dpi=150)
        plt.close()
        print("Autoencoder pipeline finished. Results in", out_dir)
    else:
        # Supervised pipeline
        if y is None:
            raise ValueError("Для supervised режима необходимо предоставить --labels")
        # determine output dim
        if args.task == 'regression':
            out_dim = 1
            loss_task = 'regression'
        else:
            # classification: detect number of classes
            classes = np.unique(y)
            out_dim = len(classes)
            loss_task = 'classification'
            # if classes not starting at 0, map
            if not np.array_equal(classes, np.arange(len(classes))):
                mapping = {v: i for i, v in enumerate(classes)}
                y_train = np.array([mapping[v] for v in y_train])
                y_val = np.array([mapping[v] for v in y_val])
                y_test = np.array([mapping[v] for v in y_test])
        input_dim = X_train.shape[1]
        hidden_dims = [max(128, input_dim*2), max(64, input_dim)]
        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=out_dim, activation='relu', dropout=0.1)
        # Datasets and loaders
        train_ds = EngineDataset(X_train, y_train, meta=[None]*len(X_train))
        val_ds = EngineDataset(X_val, y_val, meta=[None]*len(X_val))
        test_ds = EngineDataset(X_test, y_test, meta=[None]*len(X_test))
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

        print("Training supervised model...")
        model, history = train_supervised(model, train_loader, val_loader, device, epochs=args.epochs, lr=1e-3, task=loss_task, early_stop_patience=10, checkpoint_path=os.path.join(out_dir, 'super_best.pth'))
        plot_training_history(history, out_dir, title='supervised')
        preds, targets, metas_test = evaluate_supervised(model, test_loader, device, task=loss_task)

        # process outputs
        if args.task == 'regression':
            preds_flat = preds.squeeze()
            mse = mean_squared_error(targets, preds_flat)
            r2 = r2_score(targets, preds_flat)
            print(f"Test MSE: {mse:.6f}, R2: {r2:.6f}")
            df_res = pd.DataFrame({
                'path': [m['path'] if m else '' for m in metas_test],
                'target': targets,
                'pred': preds_flat,
                'abs_error': np.abs(targets - preds_flat)
            })
            save_results_table(df_res, out_dir, name='regression_results')
            # plot predicted vs true
            plt.figure()
            plt.scatter(targets, preds_flat, alpha=0.7, s=20)
            plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
            plt.xlabel('true')
            plt.ylabel('pred')
            plt.title('Prediction vs True (test)')
            plt.savefig(os.path.join(out_dir, 'pred_vs_true.png'), dpi=150)
            plt.close()
        else:
            pred_labels = preds.argmax(axis=1)
            acc = accuracy_score(targets, pred_labels)
            print(f"Test accuracy: {acc:.4f}")
            # For binary, compute AUC if possible
            auc = None
            try:
                if preds.shape[1] == 2:
                    auc = roc_auc_score(targets, preds[:,1])
                    print("AUC:", auc)
            except Exception:
                auc = None
            df_res = pd.DataFrame({
                'path': [m['path'] if m else '' for m in metas_test],
                'target': targets,
                'pred_label': pred_labels
            })
            save_results_table(df_res, out_dir, name='classification_results')
        print("Supervised pipeline finished. Results in", out_dir)

    # Optional: PCA visualization of feature space
    try:
        pca = PCA(n_components=min(3, X_scaled.shape[1]))
        X_p = pca.fit_transform(X_scaled)
        plt.figure()
        if X_p.shape[1] == 3:
            ax = plt.axes(projection='3d')
            ax.scatter(X_p[:,0], X_p[:,1], X_p[:,2], s=10)
            ax.set_title('PCA 3D of feature space')
        else:
            plt.scatter(X_p[:,0], X_p[:,1], s=10)
            plt.title('PCA 2D of feature space')
        plt.savefig(os.path.join(out_dir, 'pca_features.png'), dpi=150)
        plt.close()
    except Exception as e:
        print("PCA visualization failed:", e)

    print("Done. Результаты -- в папке:", out_dir)
    print("Рекомендации по повышению качества: увеличить объём обучающей выборки, улучшить предобработку (физические признаки), использовать ансамбли, кросс-валидацию и hyperparameter tuning (Optuna/Random Search).")

if __name__ == '__main__':
    main()
