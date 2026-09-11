"""Test-set inference with TTA, and submission.

Each fold model predicts on the test set; with TTA, predictions are averaged over the
identity plus horizontal/vertical flips. Fold models are averaged (a simple ensemble)
and optionally combined with the all-data model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import KaggleOCDataset
from .models import build_model


# TTA transforms operate on the already-normalised tensor batch.
def tta_views(x):
    """Return a list of augmented copies of batch x. Flips are label-preserving
    for cells (no canonical orientation)."""
    return [x,
            torch.flip(x, dims=[3]),          # horizontal
            torch.flip(x, dims=[2]),          # vertical
            torch.flip(x, dims=[2, 3])]       # both


def predict_with_ckpt(args, ckpt_path, test_dl, device):
    """Run TTA inference over the test set for one checkpoint -> np.array of scores."""
    model = build_model(args).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    scores = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp):
        for batch in tqdm(test_dl, desc=ckpt_path.name):
            x = batch['x'].to(device)
            views = tta_views(x) if args.use_tta else [x]
            probs = torch.zeros(x.size(0), device=device)
            for v in views:
                probs += torch.sigmoid(model(v).squeeze(1).float())
            probs /= len(views)
            scores.extend(probs.cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(scores)


def make_submission(args, norm, device):
    sub = pd.read_csv(args.data_root / 'sampleSubmission.csv')
    print('Submission shape:', sub.shape, '| Columns:', list(sub.columns))

    test_df = sub.copy()
    if 'Diagnosis' not in test_df.columns:
        test_df['Diagnosis'] = 0

    test_ds = KaggleOCDataset(
        data_root=args.data_root, df=test_df, mode=args.mode, split='test', size=args.size,
        fl_channels=args.fl_channels, bf_subdir='test', fl_subdir='test', **norm)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # --- CV fold ensemble (average of the n_folds validated models) ---
    cv_scores = np.zeros(len(test_df))
    for fold in range(args.n_folds):
        cv_scores += predict_with_ckpt(args, args.out_dir / f'fold{fold}.pt',
                                       test_dl, device) / args.n_folds

    # --- All-data model (only if it was trained) ---
    alldata_available = (args.out_dir / 'alldata.pt').exists()
    alldata_scores = (predict_with_ckpt(args, args.out_dir / 'alldata.pt', test_dl, device)
                      if alldata_available else None)

    # --- Choose what goes into the submission, per args.submission_source ---
    src = args.submission_source
    if src == 'alldata':
        assert alldata_available, "submission_source='alldata' but alldata.pt not found " \
            "-- run with --retrain-all-data."
        test_scores = alldata_scores
        print('Submission source: ALL-DATA model only.')
    elif src == 'both':
        assert alldata_available, "submission_source='both' but alldata.pt not found."
        test_scores = 0.5 * cv_scores + 0.5 * alldata_scores
        print('Submission source: average of CV ensemble + all-data model.')
    else:  # 'cv'
        test_scores = cv_scores
        print('Submission source: CV fold ensemble.')

    sub['Diagnosis'] = test_scores
    sub_path = args.out_dir / 'submission.csv'
    sub.to_csv(sub_path, index=False)
    print(f'Wrote {sub_path}')
    print(f'Score distribution: min={test_scores.min():.4f} max={test_scores.max():.4f} '
          f'mean={test_scores.mean():.4f} frac>0.5={(test_scores > 0.5).mean():.4f}')


def check_submission(args):
    """Final submission sanity check."""
    sub_check = pd.read_csv(args.out_dir / 'submission.csv')
    print('Shape:', sub_check.shape)
    print('Columns:', list(sub_check.columns))
    print('Null values:', sub_check.isna().sum().to_dict())
    assert sub_check.isna().sum().sum() == 0, 'NaNs in submission!'
    assert sub_check['Diagnosis'].between(0, 1).all(), 'Scores outside [0,1]!'
    assert len(sub_check) == len(pd.read_csv(args.data_root / 'sampleSubmission.csv')), \
        'Row count does not match sampleSubmission!'
    print('\nScore stats:')
    print(sub_check['Diagnosis'].describe())
    print('\nFirst 5 rows:')
    print(sub_check.head())
    print('\nAll checks passed - safe to submit.')
