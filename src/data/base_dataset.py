import os
import os.path as osp

import numpy as np
import cv2
from torch.utils.data import Dataset
from basicsr.utils import img2tensor


VALID_IMG_EXTS = {'.png', '.jpg', '.jpeg'}


def list_images(folder: str):
    """Return a sorted list of image paths in folder (non-recursive)."""
    return sorted(
        osp.join(folder, f) for f in os.listdir(folder)
        if osp.splitext(f)[1].lower() in VALID_IMG_EXTS
    )


def load_image_rgb(path: str):
    """Read image from disk → RGB float32 CHW tensor in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    return img2tensor(img.astype(np.float32), bgr2rgb=True, float32=True) / 255.


def normalize_clip(clip):
    """Float [0, 1] (T, C, H, W) or (N, C, H, W) → [-1, 1]."""
    mean = clip.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = clip.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    return (clip - mean) / std


class BaseImageDataset(Dataset):
    def __init__(self, data_opt, **kwargs):
        # dict to attr
        for kw, args in data_opt.items():
            setattr(self, kw, args)

        # can override options defined in data_opt
        for kw, args in kwargs.items():
            setattr(self, kw, args)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass