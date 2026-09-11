"""timm backbones with the stem patched for the input channel count, and the classifiers.

Extra (FL) channels are initialised from the mean of the pretrained RGB weights.
"""

import timm
import torch
import torch.nn as nn


def _load_simclr_encoder(model, simclr_ckpt_path):
    """Load cross-modal SimCLR encoder weights into a timm model.
    Called BEFORE the stem is patched: the SimCLR encoder was trained on 3-channel
    input, so its stem weights are 3-channel. The stem patch then copies them into the
    first 3 channels and initialises the extra channels from their mean. Loaded with
    strict=False because (a) the SimCLR projection head is intentionally absent, and
    (b) the timm classifier head of this model is intentionally not in the SimCLR checkpoint.
    """
    ckpt = torch.load(simclr_ckpt_path, map_location='cpu', weights_only=False)
    # ckpt is {'encoder': <state_dict>, 'arch': ..., 'epoch': ..., ...}
    enc_sd = ckpt['encoder'] if isinstance(ckpt, dict) and 'encoder' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    return len(missing), len(unexpected), ckpt.get('epoch', 'unknown')


_SIMCLR_REPORTED = False


def make_backbone(in_channels, out_dim=1,
                  model_name='convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True,
                  simclr_ckpt_path=None):
    """timm model with the stem patched for `in_channels`. Returns (model, feature_dim)."""
    # If using SimCLR init, we still build with ImageNet pretrained=True first --
    # the SimCLR load then overwrites the encoder weights and the stem is patched
    # afterward. This avoids a noisy cold start if something goes wrong with the SimCLR load.
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim)

    # ---- SimCLR encoder load (BEFORE stem patch) ----
    global _SIMCLR_REPORTED
    if simclr_ckpt_path is not None:
        n_missing, n_unexpected, ep = _load_simclr_encoder(model, simclr_ckpt_path)
        if not _SIMCLR_REPORTED:
            print(f'  [SimCLR init] loaded encoder from epoch {ep} | '
                  f'missing keys: {n_missing} (head + bias terms expected) | '
                  f'unexpected: {n_unexpected} (should be 0)')
            _SIMCLR_REPORTED = True

    # ConvNeXt stem is Conv2d(3, dim, k=4, s=4) then LayerNorm. Assert before patching
    # so a future timm restructure fails loudly instead of silently.
    assert hasattr(model, 'stem') and isinstance(model.stem[0], nn.Conv2d), \
        f'Unexpected stem layout for {model_name}: {getattr(model, "stem", None)}'
    old_stem = model.stem[0]
    new_stem = nn.Conv2d(in_channels, old_stem.out_channels,
                         kernel_size=old_stem.kernel_size,
                         stride=old_stem.stride,
                         padding=old_stem.padding,
                         bias=old_stem.bias is not None)
    with torch.no_grad():
        if in_channels >= 3:
            new_stem.weight[:, :3] = old_stem.weight
            if in_channels > 3:
                new_stem.weight[:, 3:] = old_stem.weight.mean(
                    dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
        else:
            new_stem.weight.copy_(old_stem.weight[:, :in_channels])
        if old_stem.bias is not None:
            new_stem.bias.copy_(old_stem.bias)
    model.stem[0] = new_stem
    return model, model.num_features


class SingleModalNet(nn.Module):
    def __init__(self, in_channels, model_name, dropout=0.3, simclr_ckpt_path=None):
        super().__init__()
        # Headless backbone + explicit head with dropout before the classifier.
        self.backbone, fdim = make_backbone(in_channels, out_dim=0, model_name=model_name,
                                            simclr_ckpt_path=simclr_ckpt_path)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fdim, 1))

    def forward(self, x):
        return self.head(self.backbone(x))


class EarlyFusionNet(nn.Module):
    def __init__(self, total_channels, model_name, dropout=0.3, simclr_ckpt_path=None):
        super().__init__()
        # Headless backbone + explicit head with dropout before the classifier.
        self.backbone, fdim = make_backbone(total_channels, out_dim=0, model_name=model_name,
                                            simclr_ckpt_path=simclr_ckpt_path)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fdim, 1))

    def forward(self, x):
        return self.head(self.backbone(x))


class LateFusionNet(nn.Module):
    def __init__(self, c_bf, c_fl, model_name, dropout=0.3, simclr_ckpt_path=None):
        super().__init__()
        self.bf, fdim = make_backbone(c_bf, out_dim=0, model_name=model_name,
                                      simclr_ckpt_path=simclr_ckpt_path)
        self.fl, _ = make_backbone(c_fl, out_dim=0, model_name=model_name,
                                   simclr_ckpt_path=simclr_ckpt_path)
        self.c_bf = c_bf
        self.head = nn.Sequential(
            nn.Linear(fdim * 2, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        bf, fl = x[:, :self.c_bf], x[:, self.c_bf:]
        return self.head(torch.cat([self.bf(bf), self.fl(fl)], dim=1))


def build_model(args):
    dp = getattr(args, 'dropout', 0.3)
    ssl = getattr(args, 'simclr_ckpt_path', None)
    if args.mode == 'BF':
        return SingleModalNet(3, args.arch, dropout=dp, simclr_ckpt_path=ssl)
    if args.mode == 'FL':
        return SingleModalNet(args.fl_channels, args.arch, dropout=dp, simclr_ckpt_path=ssl)
    total = 3 + args.fl_channels
    if args.fusion == 'E':
        return EarlyFusionNet(total, args.arch, dropout=dp, simclr_ckpt_path=ssl)
    if args.fusion == 'L':
        return LateFusionNet(3, args.fl_channels, args.arch, dropout=dp, simclr_ckpt_path=ssl)
    raise ValueError('Intermediate fusion needs CAFNet/MMTM/HcCNN from the original repo.')
