#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (512, 256, 128), p: float = 0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent: int = 64, hidden: int = 128):
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hidden, batch_first=True)
        self.enc_proj = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.dec_proj = nn.Linear(hidden, in_dim)

    def forward(self, x):
        h, _ = self.encoder(x)
        z = self.enc_proj(h)
        d, _ = self.decoder(z)
        recon = self.dec_proj(d)
        return recon, z


class JointAE_RUL(nn.Module):
    def __init__(self, in_dim: int, latent: int = 64):
        super().__init__()
        self.ae = AutoEncoder(in_dim, latent)
        self.rul_head = nn.Sequential(
            nn.Linear(latent, 128), nn.GELU(), nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        recon, z = self.ae(x)
        rul = self.rul_head(z)
        return recon, rul, z

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_cap_rul(rul: np.ndarray, cap: int = 125, k: float = 0.03) -> np.ndarray:
    return cap * np.tanh(k * rul)


def build_dataset(X: np.ndarray, y: np.ndarray, batch: int = 1024) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=True)


def train_epoch(model, loader, opt, scaler, device, task: str, alpha: float = 0.5):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            if task == 'autoencoder':
                recon, _ = model(xb)
                loss = F.mse_loss(recon, xb)
            elif task == 'joint':
                recon, rul, _ = model(xb)
                loss = alpha * F.mse_loss(recon, xb) + (1 - alpha) * F.mse_loss(rul.squeeze(-1), yb)
            else:
                out = model(xb).squeeze(-1)
                loss = F.mse_loss(out, yb) if task == 'regression' else F.binary_cross_entropy_with_logits(out, yb)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def eval_epoch(model, loader, device, task: str):
    model.eval()
    with torch.no_grad():
        if task == 'autoencoder':
            errs = []
            for xb, _ in loader:
                xb = xb.to(device)
                recon, _ = model(xb)
                errs.append(F.mse_loss(recon, xb, reduction='none').mean(1).cpu())
            e = torch.cat(errs).numpy()
            return e.mean(), e
        if task == 'joint':
            ys, ps = [], []
            for xb, yb in loader:
                xb = xb.to(device)
                _, rul, _ = model(xb)
                ps.append(rul.squeeze(-1).cpu())
                ys.append(yb)
            y = torch.cat(ys).numpy()
            p = torch.cat(ps).numpy()
            return mean_squared_error(y, p), r2_score(y, p)
        ys, ps = [], []
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).squeeze(-1)
            ps.append(out.cpu())
            ys.append(yb)
        y = torch.cat(ys).numpy()
        p = torch.cat(ps).numpy()
        if task == 'regression':
            return mean_squared_error(y, p), r2_score(y, p)
        prob = 1 / (1 + np.exp(-p))
        return accuracy_score(y, prob > 0.5), roc_auc_score(y, prob)(y, prob > 0.5), roc_auc_score(y, prob)


def fit(model, tr, va, device, task: str, epochs: int = 200):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, steps_per_epoch=len(tr), epochs=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    ema = EMA(model)
    best = float('inf')
    for _ in range(epochs):
        train_epoch(model, tr, opt, scaler, device, task)
        sch.step()
        ema.update(model)
        m1, _ = eval_epoch(model, va, device, task)
        if m1 < best:
            best = m1
            torch.save(model.state_dict(), 'best.pth')
    model.load_state_dict(torch.load('best.pth', map_location=device))
    ema.apply(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--task', type=str, choices=['regression', 'classification', 'autoencoder', 'joint'], default='regression')
    args = ap.parse_args()
    set_seed()
    df = pd.read_csv(args.data)
    y = df.iloc[:, -1].values.astype(np.float32)
    X = df.iloc[:, :-1].values.astype(np.float32)
    if args.task == 'regression':
        y = soft_cap_rul(y)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    sx = StandardScaler()
    Xtr = sx.fit_transform(Xtr)
    Xva = sx.transform(Xva)
    tr = build_dataset(Xtr, ytr)
    va = build_dataset(Xva, yva)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointAE_RUL(Xtr.shape[1]).to(device) if args.task == 'joint' else (AutoEncoder(Xtr.shape[1]).to(device) if args.task == 'autoencoder' else MLP(Xtr.shape[1], 1).to(device))(Xtr.shape[1]).to(device) if args.task == 'autoencoder' else MLP(Xtr.shape[1], 1).to(device)
    fit(model, tr, va, device, args.task)

if __name__ == '__main__':
    main()
