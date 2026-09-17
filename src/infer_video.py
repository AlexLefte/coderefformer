"""
infer_video.py
==============
Inference script for the CodeRefFormer video model (ConvGRUVideoVSR /
BidirectionalVideoVSR).

Given a folder of LQ video frames and a folder of HQ reference images,
restores the entire sequence and saves the output frames.

Usage
-----
    python infer_video.py \\
        --config   configs/train_video_codeformer_v1.yml \\
        --ckpt     experiments/my_run/ckpt/G_iter125000.pth \\
        --lq_dir   /path/to/lq_frames/          # folder with 000000.png, 000001.png, ...
        --ref_dir  /path/to/hq_references/      # HQ portrait images (same identity)
        --out_dir  /path/to/output_frames/
        --n_refs   5                             # number of references (-1 = all)
        --w        0.7                           # fidelity weight
        --tempo_extent  0                        # frames per chunk (0 = whole video)
        --seed     0

Arguments
---------
--lq_dir        Folder of LQ PNG frames, sorted alphabetically (000000.png, ...).
--ref_dir       Folder of HQ reference images for the same identity.
--out_dir       Output folder where restored frames are saved.
--n_refs        Number of reference images to use (-1 = all, default: -1).
--w             Fidelity weight in [0, 1]. 0 = quality, 1 = fidelity. (default: 0.7)
--tempo_extent  Frames per inference chunk. 0 = process the entire video at once
                (may OOM for long clips). Use e.g. 10 for long sequences.
--seed          Random seed (default: 0).
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Allow running from the project root (src/ folder)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import define_model
from utils import data_utils

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frame(path: Path) -> torch.Tensor:
    """Load a single frame as a float32 tensor in [0, 1].

    Identical to _load_image_rgb() in the training dataloader:
      cv2.imread (BGR uint8) -> img2tensor(bgr2rgb=True) -> / 255. -> (3, H, W) float32 [0,1]

    No resizing is applied — frames are loaded at their original resolution,
    same as the paired/unpaired dataset classes.
    Normalisation to [-1, 1] is done later by data_utils.pre_process_data().

    Returns:
        Tensor (3, H, W) float32 in [0, 1].
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    # img2tensor(bgr2rgb=True) equivalent: flip channels, HWC -> CHW, float32
    img = img[:, :, ::-1].astype(np.float32)   # BGR -> RGB, keep uint8 range
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0   # (3, H, W) [0, 1]
    return img


def collect_frame_paths(lq_dir: Path) -> list:
    """Return sorted list of frame paths from lq_dir."""
    paths = sorted([
        p for p in lq_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])
    if not paths:
        raise FileNotFoundError(f"No frames found in: {lq_dir}")
    return paths


def collect_ref_paths(ref_dir: Path, n_refs: int) -> list:
    """Return sorted reference image paths, up to n_refs (-1 = all)."""
    paths = sorted([
        p for p in ref_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])
    if not paths:
        raise FileNotFoundError(f"No reference images found in: {ref_dir}")
    if n_refs > 0:
        paths = paths[:n_refs]
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeRefFormer video inference — restore a face video with HQ references.")
    parser.add_argument("--config",   required=True,
                        help="Training YML config path.")
    parser.add_argument("--ckpt",     required=True,
                        help="Generator checkpoint (.pth).")
    parser.add_argument("--lq_dir",   required=True,
                        help="Folder of LQ video frames (PNG, sorted alphabetically).")
    parser.add_argument("--ref_dir",  required=True,
                        help="Folder of HQ reference images.")
    parser.add_argument("--out_dir",  default="output_frames",
                        help="Output folder for restored frames (default: output_frames).")
    parser.add_argument("--n_refs",   type=int, default=-1,
                        help="Number of references to use (-1 = all, default).")
    parser.add_argument("--w",        type=float, default=0.7,
                        help="Fidelity weight w in [0,1] (default: 0.7).")
    parser.add_argument("--tempo_extent", type=int, default=0,
                        help="Frames per inference chunk. 0 = whole video at once.")
    parser.add_argument("--seed",     type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load YML config
    with open(args.config) as f:
        opt = yaml.safe_load(f)

    # Override relevant settings
    opt['model']['generator']['w_scale'] = args.w
    opt['model']['g_load_path']  = args.ckpt
    opt['model']['d_load_path']  = None
    opt['is_train']              = False
    opt['device']                = 'cuda'

    # Stub for dataset.train — CodeFormerJointRefModel.__init__ reads it
    if 'dataset' not in opt:
        opt['dataset'] = {}
    if 'train' not in opt['dataset']:
        opt['dataset']['train'] = {}

    print(f"Building model from config: {args.config}")
    model = define_model(opt)
    model.net_G.eval()

    # Detect generator type for informational warnings.
    # ConvGRUVideoVSR, BidirectionalVideoVSR, SingleFrameVideoVSR all accept
    # the same forward() signature (B, T, C, H, W), so no code changes are needed.
    gen_type = type(model.net_G).__name__
    print(f"Generator: {gen_type}")
    if args.tempo_extent > 0 and "ConvGRU" in gen_type:
        print(
            f"[WARN] tempo_extent={args.tempo_extent} with {gen_type}: "
            "the GRU hidden state is reset at each chunk boundary, so "
            "temporal continuity is lost between chunks. "
            "Use --tempo_extent 0 to process the whole video at once "
            "if VRAM allows."
        )

    # Collect frames and references
    lq_dir     = Path(args.lq_dir)
    ref_dir    = Path(args.ref_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = collect_frame_paths(lq_dir)
    ref_paths   = collect_ref_paths(ref_dir, args.n_refs)

    print(f"LQ frames  : {len(frame_paths)} frames from {lq_dir}")
    print(f"References : {len(ref_paths)} image(s) from {ref_dir}")
    for p in ref_paths:
        print(f"  {p.name}")

    # Load all LQ frames as (T, 3, H, W) float32 in [0, 1]
    # Frames are loaded at their original resolution (no forced resize),
    # identical to the training dataloader.
    print("Loading LQ frames ...")
    lr_frames = torch.stack([load_frame(p) for p in frame_paths], dim=0)  # (T, 3, H, W)

    # Load reference images as (N, 3, H, W) float32 in [0, 1]
    ref_frames = torch.stack([load_frame(p) for p in ref_paths], dim=0)   # (N, 3, H, W)

    # Determine chunking
    T = lr_frames.shape[0]
    tempo_extent = args.tempo_extent if args.tempo_extent > 0 else T
    chunks = [
        (i, min(i + tempo_extent, T))
        for i in range(0, T, tempo_extent)
    ]
    print(f"Total frames: {T}  |  chunk size: {tempo_extent}  |  chunks: {len(chunks)}")

    # Run inference chunk by chunk
    restored_frames = []

    for chunk_start, chunk_end in tqdm(chunks, desc="Inferring chunks"):
        lr_chunk = lr_frames[chunk_start:chunk_end]    # (t, 3, H, W)

        # Add batch dimension: (1, t, 3, H, W)
        data = {
            'lr':  lr_chunk.unsqueeze(0),
            'ref': ref_frames.unsqueeze(0),   # (1, N, 3, H, W)
        }

        # Normalise to [-1, 1] — identical to data_utils.pre_process_data()
        pre_processed = data_utils.pre_process_data(data)
        data.update(pre_processed)

        # Infer
        with torch.no_grad():
            output_dict = model.infer(data)

        # hr_data is (1, t, 3, H, W) float32 in [-1, 1]
        hr_chunk = output_dict['hr_data']   # (1, t, 3, H, W)
        B, t, C, H, W = hr_chunk.shape

        # post_process() handles 4D (B, C, H, W) -> (B, H, W, C).
        # Flatten B and T into one batch dimension, process, then restore T.
        hr_flat = hr_chunk.view(B * t, C, H, W)                   # (B*t, 3, H, W)
        hr_np   = data_utils.post_process(hr_flat, bit_depth=8)   # (B*t, H, W, 3) RGB uint8
        hr_np   = hr_np.reshape(B, t, H, W, C)[0]                 # (t, H, W, 3)

        restored_frames.append(hr_np)

    # Concatenate all chunks along T
    all_frames = np.concatenate(restored_frames, axis=0)   # (T, H, W, 3) RGB uint8

    # Save each frame using save_sequence (handles to_bgr and naming)
    frame_names = [p.name for p in frame_paths]
    print(f"Saving {len(all_frames)} restored frames to: {out_dir}")
    data_utils.save_sequence(
        str(out_dir),
        all_frames,
        frm_idx_lst = frame_names,
        to_bgr      = True,    # RGB -> BGR for cv2
        dtype       = np.uint8,
    )

    print(f"Done. Restored frames saved to: {out_dir}")


if __name__ == "__main__":
    main()