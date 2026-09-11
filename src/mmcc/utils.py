import random

import numpy as np
import torch


def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
