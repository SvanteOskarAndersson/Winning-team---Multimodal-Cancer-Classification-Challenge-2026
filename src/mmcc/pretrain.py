"""Cross-modal SimCLR self-supervised pretraining (stage 1).

Pretrains a ConvNeXtV2 encoder on all available cells -- train and the unlabelled test
set -- using no labels. For each cell the positive pair is its BF view and its FL view,
so the encoder learns to map a cell's two modalities into the same region of feature
space. Each view is a single 3-channel image, so the resulting encoder still drops into
the BF, FL or early-fusion model.

Output: `<out_dir>/simclr_convnextv2_encoder.pt`, consumed by `mmcc train --simclr-ckpt`.
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from .utils import seed_everything, setup_device

# Leave headroom before Kaggle's hard runtime limit so the final save completes.
TIME_BUDGET_SEC = 11.0 * 3600


# ---------------------------------------------------------------------------
# Data: the unlabelled image pool
# ---------------------------------------------------------------------------

def check_data_root(data_root):
    assert data_root.exists(), f'{data_root} not found'
    for sub in ['BF/train', 'BF/test', 'FL/train', 'FL/test']:
        p = data_root / sub
        assert p.exists(), f'missing {p}'
        print(f'{sub:10} {len(list(p.iterdir()))} files')


def build_cell_pairs(data_root):
    # Each 'cell' is a (BF_path, FL_path) pair sharing the same filename. Pool train + test.
    cell_pairs = []
    for split in ['train', 'test']:
        bf_dir = data_root / 'BF' / split
        fl_dir = data_root / 'FL' / split
        for bf_path in sorted(bf_dir.glob('*.jpg')):
            fl_path = fl_dir / bf_path.name
            assert fl_path.exists(), f'No matching FL image for {bf_path.name}'
            cell_pairs.append((bf_path, fl_path))

    print(f'Total cells (BF+FL pairs) for cross-modal pretraining: {len(cell_pairs):,}')
    print('Examples:')
    for bf, fl in cell_pairs[:2]:
        print(f'  BF {bf}\n  FL {fl}')
    return cell_pairs


# ---------------------------------------------------------------------------
# Augmentation
#
# The two views of the positive pair are the cell's BF image and its FL image, each
# independently and strongly augmented. Cells have no canonical orientation, so
# flips/rotations are label-safe.
# ---------------------------------------------------------------------------

# Normalization: generic stats for mixed BF+FL 3-channel images.
def quick_norm_stats(paths, n=3000):
    rng = np.random.default_rng(0)
    sample = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    s = np.zeros(3); sq = np.zeros(3); count = 0
    for idx in tqdm(sample, desc='SSL norm stats'):
        a = np.asarray(Image.open(paths[idx]).convert('RGB'), np.float32) / 255.0
        s += a.reshape(-1, 3).sum(0); sq += (a.reshape(-1, 3) ** 2).sum(0)
        count += a.shape[0] * a.shape[1]
    mean = s / count
    std = np.sqrt(np.maximum(sq / count - mean ** 2, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())


def build_ssl_aug(size, ssl_mean, ssl_std):
    # SimCLR-style strong augmentation pipeline
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.4, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        # Hue=0 and no RandomGrayscale: stain color encodes diagnostic structure on
        # cytology data. Brightness/contrast/saturation are kept -- they mimic exposure
        # variation the encoder should ignore.
        transforms.RandomApply(
            [transforms.ColorJitter(0.6, 0.6, 0.6, 0)], p=0.8),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(ssl_mean, ssl_std),
    ])


class CrossModalDataset(Dataset):
    """For each cell, returns an augmented BF view and an augmented FL view.
    These form the SimCLR positive pair -- the encoder learns to map a cell's two
    modalities to the same place in feature space. No labels."""
    def __init__(self, cell_pairs, transform):
        self.cell_pairs = cell_pairs
        self.transform = transform

    def __len__(self):
        return len(self.cell_pairs)

    def __getitem__(self, idx):
        bf_path, fl_path = self.cell_pairs[idx]
        bf = Image.open(bf_path).convert('RGB')
        fl = Image.open(fl_path).convert('RGB')
        # independent strong augmentation on each modality
        return self.transform(bf), self.transform(fl)


# ---------------------------------------------------------------------------
# Model: encoder + projection head
#
# The encoder is the ConvNeXtV2 backbone whose weights we keep. The projection head is a
# small MLP used only during pretraining; the loss is computed on its output, then it is
# discarded.
# ---------------------------------------------------------------------------

class SimCLRNet(nn.Module):
    def __init__(self, arch, proj_dim=128, pretrained=True):
        super().__init__()
        # num_classes=0 -> backbone outputs a feature vector, no classifier head
        self.encoder = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        feat_dim = self.encoder.num_features
        self.feat_dim = feat_dim
        # 2-layer MLP projection head (SimCLR v1/v2 standard)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)              # representation (kept after pretraining)
        z = self.projector(h)            # projection (used only for the loss)
        return F.normalize(z, dim=1)     # L2-normalised for cosine similarity


# ---------------------------------------------------------------------------
# NT-Xent contrastive loss
#
# For a batch of N cells there are 2N views. Each view's positive partner is the other
# view of the same cell; all other 2N-2 views are negatives.
# ---------------------------------------------------------------------------

def nt_xent_loss(z1, z2, temperature):
    """z1, z2: (N, proj_dim) L2-normalised projections of the two views."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # (2N, d)
    sim = z @ z.t() / temperature                  # (2N, 2N) cosine sim / T
    # mask out self-similarity on the diagonal
    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float('-inf'))
    # positive index: view i's partner is i+N (and vice versa)
    targets = torch.arange(2 * N, device=z.device)
    targets = (targets + N) % (2 * N)
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(cfg):
    check_data_root(cfg.data_root)
    cell_pairs = build_cell_pairs(cfg.data_root)
    # A flat list of every image path, used only for computing normalization stats.
    all_image_paths = [p for pair in cell_pairs for p in pair]

    device = setup_device()
    cfg.out_dir.mkdir(exist_ok=True, parents=True)
    seed_everything(cfg.seed)

    ssl_mean, ssl_std = quick_norm_stats(all_image_paths)
    print(f'SSL mean: {ssl_mean}\nSSL std:  {ssl_std}')
    ssl_aug = build_ssl_aug(cfg.size, ssl_mean, ssl_std)

    ssl_ds = CrossModalDataset(cell_pairs, ssl_aug)
    ssl_dl = DataLoader(ssl_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=False, prefetch_factor=4)
    print(f'{len(ssl_ds):,} cells | {len(ssl_dl)} batches/epoch at batch_size={cfg.batch_size}')
    print('Positive pair = (augmented BF view, augmented FL view) of the same cell.')

    # Starting from ImageNet weights speeds convergence; SimCLR then
    # re-shapes them around cell morphology.
    model = SimCLRNet(cfg.arch, proj_dim=cfg.proj_dim, pretrained=True).to(device)
    print(f'Encoder feature dim: {model.feat_dim}')
    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

    # smoke test
    _z = model(torch.randn(4, 3, cfg.size, cfg.size).to(device))
    print('Projection output shape:', tuple(_z.shape))

    # quick numerical sanity check
    _z1 = F.normalize(torch.randn(8, cfg.proj_dim), dim=1)
    _z2 = F.normalize(torch.randn(8, cfg.proj_dim), dim=1)
    print('NT-Xent on random batch (should be ~log(2N-1)):',
          f'{nt_xent_loss(_z1, _z2, cfg.temperature).item():.3f}',
          f'| reference log(15)={np.log(15):.3f}')

    # --- Optimizer, schedule, and resume logic ---
    # AdamW with a linear warmup then cosine decay. The training state is checkpointed
    # every epoch; if a run times out, re-running the same command resumes.
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    def lr_at(epoch):
        """Linear warmup then cosine decay, as a multiplier on cfg.lr."""
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    start_epoch = 0
    loss_history = []
    resume_path = cfg.out_dir / cfg.resume_name
    if resume_path.exists():
        # weights_only=False: the scheduler state holds numpy floats from lr_at, which
        # torch>=2.6's default weights_only=True loader rejects.
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])
        print(f'Resumed from epoch {start_epoch} (found {resume_path.name}).')
    else:
        print('No resume checkpoint - starting SimCLR pretraining from scratch.')

    # --- Pretraining loop ---
    # After every epoch, save both the resume checkpoint (full state) and the
    # encoder-only checkpoint (what supervised training consumes).
    encoder_path = cfg.out_dir / cfg.ckpt_name
    t_start = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(ssl_dl, desc=f'epoch {epoch:03d}', leave=False)
        for bf_view, fl_view in pbar:
            v1 = bf_view.to(device, non_blocking=True)
            v2 = fl_view.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                z1 = model(v1)
                z2 = model(v2)
                loss = nt_xent_loss(z1, z2, cfg.temperature)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        sched.step()
        epoch_loss = running / len(ssl_dl)
        loss_history.append(epoch_loss)
        print(f'epoch {epoch:03d} | NT-Xent loss {epoch_loss:.4f} '
              f'| lr {opt.param_groups[0]["lr"]:.2e}')

        # (a) full state for resuming
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch, 'loss_history': loss_history},
                   cfg.out_dir / cfg.resume_name)
        # (b) encoder-only weights for supervised fine-tuning
        torch.save({'encoder': model.encoder.state_dict(),
                    'arch': cfg.arch, 'epoch': epoch,
                    'ssl_mean': ssl_mean, 'ssl_std': ssl_std},
                   encoder_path)

        if time.time() - t_start > TIME_BUDGET_SEC:
            print(f'\nTime budget reached at epoch {epoch}. '
                  f'Encoder saved. Re-run the same command to resume.')
            break

    print(f'\nDone. Encoder checkpoint: {encoder_path}')

    plot_loss_curve(loss_history, cfg.out_dir / 'simclr_loss_curve.png')
    verify_encoder(encoder_path)

    for f in sorted(cfg.out_dir.glob('*.pt')):
        print(f'{os.path.getsize(f) / 1e6:.1f} MB  {f.name}')


def plot_loss_curve(loss_history, path):
    """Loss curve -- did the encoder actually learn?"""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, lw=1.5)
    plt.xlabel('epoch'); plt.ylabel('NT-Xent loss')
    plt.title('SimCLR pretraining loss')
    plt.grid(alpha=0.3)
    plt.savefig(path)
    plt.close()
    print(f'Saved loss curve -> {path}')

    if len(loss_history) >= 2:
        print(f'First epoch loss: {loss_history[0]:.4f}')
        print(f'Last  epoch loss: {loss_history[-1]:.4f}')
        print('A steadily decreasing curve means the encoder is learning to tell cells apart.')
        print('A flat curve near log(2N-1) means it has not learned - check augmentation/lr.')


def verify_encoder(encoder_path):
    """Mirror how supervised training consumes the checkpoint: build a fresh timm
    ConvNeXtV2 and load the pretrained encoder weights into it."""
    ckpt = torch.load(encoder_path, map_location='cpu')
    print(f'Checkpoint arch: {ckpt["arch"]} | saved at epoch {ckpt["epoch"]}')

    verify = timm.create_model(ckpt['arch'], pretrained=False, num_classes=0)
    missing, unexpected = verify.load_state_dict(ckpt['encoder'], strict=False)
    print(f'Loaded encoder weights | missing keys: {len(missing)} | unexpected: {len(unexpected)}')
    assert len(unexpected) == 0, f'Unexpected keys - arch mismatch: {unexpected[:5]}'
    print('\nEncoder checkpoint is valid and ready for supervised fine-tuning '
          '(mmcc train --simclr-ckpt).')
