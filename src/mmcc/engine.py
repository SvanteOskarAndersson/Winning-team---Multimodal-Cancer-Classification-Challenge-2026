"""Training utilities: mixup, weighted BCE, epoch loop, metrics."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm.auto import tqdm

from .data import patient_id


def mixup(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def weighted_bce(logits, y, w_pos, w_neg):
    w = torch.where(y > 0.5, torch.full_like(y, w_pos), torch.full_like(y, w_neg))
    return F.binary_cross_entropy_with_logits(logits, y, weight=w)


def run_epoch(model, loader, optimizer, scaler, train_mode,
              use_mixup, use_amp, device, w_pos, w_neg, mixup_alpha=0.4):
    model.train(train_mode)
    all_y, all_s, all_names = [], [], []
    total_loss = 0.0
    pbar = tqdm(loader, desc='train' if train_mode else 'eval', leave=False)
    for batch in pbar:
        x = batch['x'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True).float()
        if train_mode and use_mixup:
            x, y_a, y_b, lam = mixup(x, y, alpha=mixup_alpha)
        with torch.set_grad_enabled(train_mode):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x).squeeze(1)
                if train_mode and use_mixup:
                    loss = (lam * weighted_bce(logits, y_a, w_pos, w_neg)
                            + (1 - lam) * weighted_bce(logits, y_b, w_pos, w_neg))
                else:
                    loss = weighted_bce(logits, y, w_pos, w_neg)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0)
        all_s.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
        all_y.extend(y.detach().cpu().tolist())
        all_names.extend(batch['name'])
    return (total_loss / len(loader.dataset),
            np.array(all_y), np.array(all_s), all_names)


def cell_metrics(y, s, threshold=0.5):
    p = (s > threshold).astype(int)
    return {
        'AUC':  metrics.roc_auc_score(y, s) if len(np.unique(y)) > 1 else float('nan'),
        'F1':   metrics.f1_score(y, p, zero_division=0),
        'Acc':  metrics.accuracy_score(y, p),
        'Prec': metrics.precision_score(y, p, zero_division=0),
        'Rec':  metrics.recall_score(y, p, zero_division=0),
    }


def patient_auc(names, y, s):
    """Aggregate cell scores to patient level (mean) and compute AUC.
    This matches the weak-label structure of the task."""
    d = pd.DataFrame({'patient': [patient_id(n) for n in names], 'y': y, 's': s})
    g = d.groupby('patient').agg(y=('y', 'first'), s=('s', 'mean'))
    if g['y'].nunique() < 2:
        return float('nan')
    return metrics.roc_auc_score(g['y'], g['s'])
