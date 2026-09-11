"""Data layout checks, patient-aware folds, normalization statistics and the dataset."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

def check_data_root(data_root):
    data_root = Path(data_root)
    assert data_root.exists(), f'{data_root} not found'
    assert (data_root / 'train.csv').exists(), 'train.csv missing'
    assert (data_root / 'BF' / 'train').exists(), 'BF/train missing'
    assert (data_root / 'FL' / 'train').exists(), 'FL/train missing'
    print('Data layout OK.')
    print('Sample BF file:', next((data_root / 'BF' / 'train').iterdir()))


# ---------------------------------------------------------------------------
# FL channel probe
#
# This determines whether the autofluorescence signal is used. The LetItShine paper's
# central finding is that fluorescence carries extra diagnostic signal. If the FL files
# are 4-channel, loading them as RGB throws that channel away, so the channel count is
# read from the real files.
# ---------------------------------------------------------------------------

def probe_modality(folder, n=8):
    modes, bands, sizes = set(), set(), set()
    for p in list(sorted(folder.iterdir()))[:n]:
        im = Image.open(p)
        modes.add(im.mode)
        bands.add(im.getbands())
        sizes.add(im.size)
    return modes, bands, sizes


def detect_fl_channels(data_root):
    data_root = Path(data_root)
    bf_modes, bf_bands, bf_sizes = probe_modality(data_root / 'BF' / 'train')
    fl_modes, fl_bands, fl_sizes = probe_modality(data_root / 'FL' / 'train')
    print('BF  modes:', bf_modes, '| bands:', bf_bands, '| sizes:', bf_sizes)
    print('FL  modes:', fl_modes, '| bands:', fl_bands, '| sizes:', fl_sizes)

    # Decide FL channel count from what the files actually contain.
    _fl_band_lens = {len(b) for b in fl_bands}
    assert len(_fl_band_lens) == 1, f'Inconsistent FL channel counts across files: {fl_bands}'
    fl_channels = _fl_band_lens.pop()
    assert fl_channels in (3, 4), f'Unexpected FL channel count: {fl_channels}'
    print(f'\n>>> FL_CHANNELS = {fl_channels}'
          + ('  (4-channel: autofluorescence channel WILL be kept)' if fl_channels == 4
             else '  (3-channel RGB)'))
    return fl_channels


# ---------------------------------------------------------------------------
# Patient-aware fold splits
#
# Filenames look like `pat_NNN_*.jpg`. We group by patient ID so no patient leaks across
# train/val, and stratify so every fold has a comparable cancer rate -- important with
# few patients, where a plain GroupKFold can produce a fold with almost no cancer patients.
# ---------------------------------------------------------------------------

PAT_RE = re.compile(r'pat_(\d+)')


def patient_id(name: str) -> str:
    m = PAT_RE.search(name)
    if m is None:
        raise ValueError(f'No patient ID in filename: {name}')
    return m.group(1)


def make_folds(train_csv, n_folds=3, seed=0):
    df = pd.read_csv(train_csv)
    df['patient'] = df['Name'].map(patient_id)
    df['fold'] = -1
    # StratifiedGroupKFold: groups (patients) never split, classes kept balanced per fold.
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(
            sgkf.split(df, df['Diagnosis'], df['patient'])):
        df.loc[val_idx, 'fold'] = fold_idx
    assert (df['fold'] >= 0).all(), 'Some rows were not assigned a fold'
    return df


def print_fold_summary(full_df, n_folds):
    print(f'Total rows: {len(full_df)}')
    print(f'Total patients: {full_df["patient"].nunique()}')
    print(f'Patients per fold: {full_df.groupby("fold")["patient"].nunique().to_dict()}')
    print(f'Cancer rate (cells) per fold: '
          f'{full_df.groupby("fold")["Diagnosis"].mean().round(3).to_dict()}')

    # Patient-level cancer balance per fold -- watch for a degenerate fold.
    _pat = full_df.groupby('patient').agg(fold=('fold', 'first'), label=('Diagnosis', 'first'))
    print(f'Cancer patients per fold: '
          f'{_pat.groupby("fold")["label"].sum().astype(int).to_dict()}')
    print(f'Total patients per fold:  '
          f'{_pat.groupby("fold")["label"].count().to_dict()}')
    for f in range(n_folds):
        sub = _pat[_pat.fold == f]
        if sub['label'].nunique() < 2:
            print(f'  WARNING: fold {f} has only one class at patient level -- '
                  f'its AUC will be meaningless.')


def class_weights(full_df):
    # Class weights from the actual training distribution (replaces the paper's hardcoded
    # 0.78/0.22, which were specific to their dataset).
    pos_rate = full_df['Diagnosis'].mean()
    neg_rate = 1 - pos_rate
    # Inverse-frequency: rare class weighted up.
    w_pos = float(neg_rate)
    w_neg = float(pos_rate)
    print(f'Cancer rate (cells): {pos_rate:.4f} | Healthy rate: {neg_rate:.4f}')
    print(f'Class weights -> W_POS={w_pos:.4f}, W_NEG={w_neg:.4f}')
    return w_pos, w_neg


# ---------------------------------------------------------------------------
# Normalization statistics
#
# Computed from a random sample of training images (the paper's hardcoded stats were for
# a different scanner). FL stats have the correct number of channels.
# ---------------------------------------------------------------------------

def compute_norm_stats(data_root, names, subdir='train', modality='BF',
                       fl_channels=3, sample_size=2000):
    rng = np.random.default_rng(0)
    sample = rng.choice(names, size=min(sample_size, len(names)), replace=False)
    folder = 'BF' if modality == 'BF' else 'FL'
    n_channels = 3 if modality == 'BF' else fl_channels
    target_mode = 'RGBA' if (modality == 'FL' and fl_channels == 4) else 'RGB'
    sums = np.zeros(n_channels, dtype=np.float64)
    sqs = np.zeros(n_channels, dtype=np.float64)
    count = 0
    for nm in tqdm(sample, desc=f'Norm stats {modality}'):
        path = Path(data_root) / folder / subdir / nm
        arr = np.asarray(Image.open(path).convert(target_mode), dtype=np.float32) / 255.0
        sums += arr.reshape(-1, n_channels).sum(axis=0)
        sqs += (arr.reshape(-1, n_channels) ** 2).sum(axis=0)
        count += arr.shape[0] * arr.shape[1]
    mean = sums / count
    std = np.sqrt(np.maximum(sqs / count - mean ** 2, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())


# ---------------------------------------------------------------------------
# Dataset
#
# Photometric augmentation is per-modality and train-only. The geometric stage (flips,
# rotation, affine, erasing) runs on the *concatenated* tensor so BF and FL receive the
# identical spatial transform and stay aligned. Eval/test use clean transforms.
# ---------------------------------------------------------------------------

class KaggleOCDataset(Dataset):
    def __init__(self, data_root, df, mode='MM', split='train', size=128,
                 bf_mean=None, bf_std=None, fl_mean=None, fl_std=None,
                 fl_channels=3, bf_subdir='train', fl_subdir='train'):
        assert split in ('train', 'val', 'test'), f'bad split: {split}'
        self.root = Path(data_root)
        self.df = df.reset_index(drop=True)
        self.mode, self.split, self.size = mode, split, size
        self.fl_channels = fl_channels
        self.bf_subdir, self.fl_subdir = bf_subdir, fl_subdir
        self.fl_mode = 'RGBA' if fl_channels == 4 else 'RGB'

        if split == 'train':
            # BF photometric: strong stain/illumination jitter to break the
            # patient-identity shortcut -- staining/lighting is what separates the patients.
            self.tf_bf = transforms.Compose([
                transforms.RandomChoice(
                    [transforms.RandomPosterize(3, p=1.0),
                     transforms.GaussianBlur(5, 1.5),
                     transforms.RandomSolarize(100, p=1.0)],
                    p=[0.4, 0.2, 0.4]),
                transforms.ColorJitter(brightness=0.6, contrast=0.4,
                                       saturation=0.4, hue=0),
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(bf_mean, bf_std),
            ])
            # FL photometric: sigma up to 3.2 is already near the range that destroys
            # fine nuclear texture (the MAC signal).
            # ColorJitter on a 4-channel image is invalid; only jitter the RGB part.
            fl_aug = ([transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                              saturation=0.8, hue=0.1)]
                      if fl_channels == 3 else [])
            self.tf_fl = transforms.Compose(fl_aug + [
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.GaussianBlur(5, sigma=(0.3, 3.2)),
                transforms.Normalize(fl_mean, fl_std),
            ])
            # Geometric stage runs on the CONCATENATED tensor, so BF and FL receive the
            # SAME spatial transform and stay aligned.
            # Cells have no canonical orientation, so these are label-preserving.
            self.tf_geom = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08),
                                        scale=(0.9, 1.1), shear=8),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ])
        else:  # 'val' or 'test'
            self.tf_bf = transforms.Compose([
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(bf_mean, bf_std),
            ])
            self.tf_fl = transforms.Compose([
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(fl_mean, fl_std),
            ])
            self.tf_geom = None

    def __len__(self):
        return len(self.df)

    def _load_bf(self, name):
        return Image.open(self.root / 'BF' / self.bf_subdir / name).convert('RGB')

    def _load_fl(self, name):
        return Image.open(self.root / 'FL' / self.fl_subdir / name).convert(self.fl_mode)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['Name']
        label = float(row['Diagnosis']) if 'Diagnosis' in row else -1.0
        bf = fl = None
        if self.mode in ('BF', 'MM'):
            bf = self.tf_bf(self._load_bf(name))
        if self.mode in ('FL', 'MM'):
            fl = self.tf_fl(self._load_fl(name))
        if self.mode == 'MM':
            x = torch.cat([bf, fl], dim=0)
        else:
            x = bf if self.mode == 'BF' else fl
        if self.tf_geom is not None:
            x = self.tf_geom(x)
        return {'x': x, 'label': label, 'name': name}
