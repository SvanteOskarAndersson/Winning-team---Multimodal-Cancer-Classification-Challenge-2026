"""Command-line interface.

    mmcc pretrain --data-root DATA [options]   cross-modal SimCLR pretraining
    mmcc train    --data-root DATA [options]   supervised CV + all-data retrain + submission

Defaults match the configuration used in the competition runs.
"""

import argparse
from pathlib import Path

ARCH = 'convnextv2_tiny.fcmae_ft_in22k_in1k'


def build_parser():
    parser = argparse.ArgumentParser(prog='mmcc', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    flag = argparse.BooleanOptionalAction

    # ---- pretrain -----------------------------------------------------------------
    p = sub.add_parser('pretrain', help='Cross-modal SimCLR pretraining (no labels)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-root', type=Path, required=True, help='Competition data directory')
    p.add_argument('--out-dir', type=Path, default=Path('outputs/pretrain'))
    p.add_argument('--arch', default=ARCH)
    p.add_argument('--size', type=int, default=128, help='native cell resolution')
    p.add_argument('--proj-dim', type=int, default=128, help='projection head output dim')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=5e-4, help='cosine-decayed')
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.2, help='NT-Xent temperature')
    p.add_argument('--warmup-epochs', type=int, default=10)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--amp', dest='use_amp', action=flag, default=True)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(ckpt_name='simclr_convnextv2_encoder.pt',
                   resume_name='simclr_resume.pt')  # full training state for resuming

    # ---- train ----------------------------------------------------------------------
    t = sub.add_parser('train', help='Supervised CV training, all-data retrain and submission',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument('--data-root', type=Path, required=True, help='Competition data directory')
    t.add_argument('--out-dir', type=Path, default=Path('outputs/train'))
    t.add_argument('--mode', default='MM', choices=['BF', 'FL', 'MM'])
    t.add_argument('--fusion', default='E', choices=['E', 'L'],
                   help='early or late fusion, only for MM')
    t.add_argument('--arch', default=ARCH)
    t.add_argument('--size', type=int, default=128, help='native resolution')
    t.add_argument('--epochs', type=int, default=16)
    t.add_argument('--batch-size', type=int, default=128, help='drop to 64 if OOM')
    t.add_argument('--lr', type=float, default=4e-5)
    t.add_argument('--wd', type=float, default=0.20)
    t.add_argument('--dropout', type=float, default=0.15, help='dropout before the head')
    t.add_argument('--n-folds', type=int, default=2)
    t.add_argument('--mixup', dest='use_mixup', action=flag, default=True)
    t.add_argument('--mixup-alpha', type=float, default=0.4)
    t.add_argument('--amp', dest='use_amp', action=flag, default=True)
    t.add_argument('--tta', dest='use_tta', action=flag, default=True,
                   help='test-time augmentation at inference')
    t.add_argument('--num-workers', type=int, default=4)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--retrain-all-data', action=flag, default=True,
                   help='also train one model on all labelled patients')
    t.add_argument('--retrain-epochs', type=int, default=None,
                   help='epochs for the all-data model (default: --epochs)')
    t.add_argument('--submission-source', default='both', choices=['cv', 'alldata', 'both'],
                   help='fold ensemble, all-data model, or the average of the two')
    t.add_argument('--simclr-ckpt', dest='simclr_ckpt_path', type=Path, default=None,
                   help='SimCLR encoder checkpoint from `mmcc pretrain` (default: ImageNet init)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'pretrain':
        from .pretrain import run
    else:
        from .train import run
    run(args)


if __name__ == '__main__':
    main()
