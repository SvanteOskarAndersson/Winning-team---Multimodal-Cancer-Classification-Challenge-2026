"""Supervised training: patient-aware CV, all-data retrain, threshold tuning, submission.

Pass `--simclr-ckpt` to initialise the encoder from the cross-modal SimCLR checkpoint
produced by `mmcc pretrain` instead of ImageNet weights.
"""

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import (KaggleOCDataset, check_data_root, class_weights, compute_norm_stats,
                   detect_fl_channels, make_folds, print_fold_summary)
from .engine import cell_metrics, patient_auc, run_epoch
from .inference import check_submission, make_submission
from .models import build_model
from .utils import seed_everything, setup_device


def model_smoke_test(args, device):
    _m = build_model(args).to(device)
    _in_c = (3 + args.fl_channels if args.mode == 'MM'
             else 3 if args.mode == 'BF' else args.fl_channels)
    _x = torch.randn(2, _in_c, args.size, args.size).to(device)
    print('Forward pass shape:', _m(_x).shape)
    print(f'Total params: {sum(p.numel() for p in _m.parameters()):,}')
    del _m, _x
    torch.cuda.empty_cache()


def train_folds(args, full_df, norm, device, w_pos, w_neg):
    """Train across folds. Checkpoint is selected on validation AUC (the competition
    metric), not F1. Cell-level and patient-level AUC are logged each epoch."""
    fold_oof = np.zeros(len(full_df))
    fold_summary = []
    loss_kw = dict(device=device, w_pos=w_pos, w_neg=w_neg)

    for fold in range(args.n_folds):
        print(f'\n========= FOLD {fold} =========')
        tr_df = full_df[full_df.fold != fold]
        va_df = full_df[full_df.fold == fold]

        common = dict(data_root=args.data_root, mode=args.mode, size=args.size,
                      fl_channels=args.fl_channels, **norm)
        train_ds = KaggleOCDataset(df=tr_df, split='train', **common)
        val_ds = KaggleOCDataset(df=va_df, split='val', **common)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=False, prefetch_factor=4)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=False, prefetch_factor=4)

        model = build_model(args).to(device)
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-7)
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        best_smoothed = -1.0
        auc_history = []          # raw per-epoch validation cell-AUC
        ckpt_path = args.out_dir / f'fold{fold}.pt'
        for epoch in range(args.epochs):
            tr_loss, _, _, _ = run_epoch(model, train_dl, opt, scaler, train_mode=True,
                                         use_mixup=args.use_mixup, use_amp=args.use_amp,
                                         mixup_alpha=args.mixup_alpha, **loss_kw)
            sch.step()
            va_loss, y, s, names = run_epoch(model, val_dl, opt, scaler, train_mode=False,
                                             use_mixup=False, use_amp=args.use_amp, **loss_kw)
            m = cell_metrics(y, s)
            p_auc = patient_auc(names, y, s)
            auc_history.append(m['AUC'])
            # Select on a 3-epoch moving average of validation AUC, not the single
            # highest epoch -- the raw curve is noisy and picking its peak is optimistic.
            smoothed = float(np.mean(auc_history[-3:]))
            print(f'  epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} '
                  f'| cell_AUC={m["AUC"]:.4f} (sm {smoothed:.4f}) | pat_AUC={p_auc:.4f} '
                  f'| F1={m["F1"]:.4f} Acc={m["Acc"]:.4f}')
            if smoothed > best_smoothed:
                best_smoothed = smoothed
                torch.save(model.state_dict(), ckpt_path)

        # OOF predictions from the saved (smoothed-best) checkpoint
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        _, y, s, names = run_epoch(model, val_dl, opt, scaler, train_mode=False,
                                   use_mixup=False, use_amp=args.use_amp, **loss_kw)
        fold_oof[va_df.index] = s
        f_cauc = cell_metrics(y, s)['AUC']      # actual cell-AUC of the saved checkpoint
        f_pauc = patient_auc(names, y, s)
        fold_summary.append({'fold': fold, 'ckpt_cell_AUC': f_cauc,
                             'smoothed_AUC': best_smoothed, 'pat_AUC': f_pauc})
        print(f'  Fold {fold} saved-ckpt cell AUC: {f_cauc:.4f} '
              f'(smoothed {best_smoothed:.4f}) | patient AUC: {f_pauc:.4f}')
        del model, opt, sch, scaler, train_dl, val_dl
        torch.cuda.empty_cache()

    print('\n=== Per-fold summary ===')
    print(pd.DataFrame(fold_summary).to_string(index=False))

    oof_cell = cell_metrics(full_df['Diagnosis'].values, fold_oof)
    oof_pat = patient_auc(full_df['Name'].tolist(), full_df['Diagnosis'].values, fold_oof)
    print('\n=== Out-of-fold metrics ===')
    print(f'  Cell-level AUC:    {oof_cell["AUC"]:.4f}')
    print(f'  Patient-level AUC: {oof_pat:.4f}   <-- closest to the competition target')
    for k in ('F1', 'Acc', 'Prec', 'Rec'):
        print(f'  {k}: {oof_cell[k]:.4f}')
    return fold_oof


def retrain_all_data(args, full_df, norm, device, w_pos, w_neg):
    """Retrain a single model on all labelled patients (train + val combined), trading
    the ability to validate for ~2x the training data. Runs for a fixed number of epochs
    (`retrain_epochs`, defaulting to `epochs`) since there is no validation set."""
    alldata_ckpt = args.out_dir / 'alldata.pt'
    n_ep = args.retrain_epochs if args.retrain_epochs is not None else args.epochs
    print(f'=== ALL-DATA RETRAIN: {full_df["patient"].nunique()} patients, '
          f'{len(full_df)} cells, {n_ep} epochs (no validation) ===')

    # Train on EVERY labelled cell -- no held-out fold.
    common = dict(data_root=args.data_root, mode=args.mode, size=args.size,
                  fl_channels=args.fl_channels, **norm)
    all_ds = KaggleOCDataset(df=full_df, split='train', **common)
    all_dl = DataLoader(all_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=False, prefetch_factor=4)

    model = build_model(args).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for epoch in range(n_ep):
        tr_loss, _, _, _ = run_epoch(model, all_dl, opt, scaler, train_mode=True,
                                     use_mixup=args.use_mixup, use_amp=args.use_amp,
                                     mixup_alpha=args.mixup_alpha,
                                     device=device, w_pos=w_pos, w_neg=w_neg)
        sch.step()
        # No validation set -> only training loss is available to watch.
        print(f'  [all-data] epoch {epoch:02d} | tr_loss {tr_loss:.4f} '
              f'| lr {opt.param_groups[0]["lr"]:.2e}')

    # Always save the FINAL-epoch weights: with a cosine schedule the last epoch is the
    # fully-annealed model, and we have no val signal to prefer any earlier epoch.
    torch.save(model.state_dict(), alldata_ckpt)
    print(f'Saved all-data model -> {alldata_ckpt}')
    del model, opt, sch, scaler, all_dl
    torch.cuda.empty_cache()


def tune_threshold(full_df, fold_oof):
    """Threshold tuning for the F1/accuracy diagnostics only (AUC needs no threshold)."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.1, 0.9, 0.02):
        f1 = metrics.f1_score(full_df['Diagnosis'].values,
                              (fold_oof > thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f'Best OOF F1 {best_f1:.4f} at threshold {best_thr:.2f}')


def run(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)
    check_data_root(args.data_root)
    args.fl_channels = detect_fl_channels(args.data_root)

    device = setup_device()
    seed_everything(args.seed)

    full_df = make_folds(args.data_root / 'train.csv', n_folds=args.n_folds, seed=args.seed)
    print_fold_summary(full_df, args.n_folds)
    w_pos, w_neg = class_weights(full_df)

    bf_mean, bf_std = compute_norm_stats(args.data_root, full_df['Name'].tolist(),
                                         modality='BF')
    fl_mean, fl_std = compute_norm_stats(args.data_root, full_df['Name'].tolist(),
                                         modality='FL', fl_channels=args.fl_channels)
    print(f'BF mean: {bf_mean}\nBF std:  {bf_std}')
    print(f'FL mean: {fl_mean}\nFL std:  {fl_std}')
    assert len(fl_mean) == args.fl_channels, 'FL stat channel count mismatch'
    norm = dict(bf_mean=bf_mean, bf_std=bf_std, fl_mean=fl_mean, fl_std=fl_std)

    model_smoke_test(args, device)

    fold_oof = train_folds(args, full_df, norm, device, w_pos, w_neg)

    if args.retrain_all_data:
        retrain_all_data(args, full_df, norm, device, w_pos, w_neg)
    else:
        print('retrain_all_data = False -> skipping all-data retrain (using CV fold models).')

    tune_threshold(full_df, fold_oof)

    make_submission(args, norm, device)
    check_submission(args)
