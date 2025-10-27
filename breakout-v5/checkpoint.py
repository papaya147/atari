import os
import glob
import re
import torch

def new(dir, v, d):
    filepath = os.path.join(dir, f'checkpoint_{v}.pt')
    torch.save(d, filepath)

def load(dir, v):
    filepath = os.path.join(dir, f'checkpoint_{v}.pt')
    return torch.load(filepath)

def load_latest(dir):
    pattern = os.path.join(dir, 'checkpoint_*.pt')
    max_v = -1

    for filepath in glob.glob(pattern):
        match = re.search(r'checkpoint_(\d+)\.pt$', os.path.basename(filepath))
        if match:
            v = int(match.group(1))
            max_v = max(max_v, v)

    if max_v == -1:
        return None
    
    return load(dir, max_v)
