import os
import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(model: torch.nn.Module, path: str, **kwargs):
    meta = {'model_state_dict': model.state_dict(), **kwargs}
    torch.save(meta, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def compute_confidence(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return conf, pred

def top_k_indices(scores: torch.Tensor, k: int):
    if k >= scores.size(0):
        return torch.arange(scores.size(0))
    _, idx = torch.topk(scores, k)
    return idx

def load_images_from_folder(folder: str, ext: tuple = ('png','jpg','jpeg')):
    files = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(ext):
            files.append(os.path.join(folder, fname))
    return files

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
