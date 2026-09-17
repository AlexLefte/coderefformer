"""
infer.py
========
Quick inference script for CodeRefFormer (V1 / V3 / V5).

Given a single LQ image and a folder of HQ reference images, restores the
face and saves the result.

Usage
-----
    python infer.py \\
        --config  configs/train_coderefformer_v1.yml \\
        --ckpt    experiments/my_run/ckpt/G_iter125000.pth \\
        --lq      /path/to/lq_face.png \\
        --ref_dir /path/to/reference_folder \\
        --n_refs  4 \\
        --output  result.png \\
        --w       0.7

Arguments
---------
--config    Training YML config (used to build the model).
--ckpt      Generator checkpoint (.pth).
--lq        Path to the low-quality input image.
--ref_dir   Folder containing HQ reference images for the same identity.
--n_refs    How many references to use (default: all available, -1).
            If fewer exist than requested, all are used.
            References are sorted alphabetically and the first N are taken.
--output    Where to save the restored image (default: result.png).
--w         Fidelity weight w in [0, 1].  0 = max quality, 1 = max fidelity.
            (default: 0.7)
--seed      Random seed for reproducibility (default: 0).
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from utils import data_utils

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import define_model

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_tensor(path: Path, size: int = 512) -> torch.Tensor:
    """Load an image from disk, resize to size×size, normalise to [-1, 1].

    Returns:
        Tensor of shape (1, 3, size, size) in float32.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32) / 255.0          # [0, 1]
    img = img * 2.0 - 1.0                          # [-1, 1]
    t   = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)
    return t.unsqueeze(0)                           # (1, 3, H, W)


def collect_ref_paths(ref_dir: Path, n_refs: int) -> list:
    """Collect reference image paths from ref_dir, sorted alphabetically.

    Args:
        ref_dir: directory containing HQ reference images.
        n_refs:  maximum number of references to use (-1 = all).

    Returns:
        list of Path objects.
    """
    paths = sorted([
        p for p in ref_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])
    if not paths:
        raise FileNotFoundError(f"No images found in reference folder: {ref_dir}")
    if n_refs > 0:
        paths = paths[:n_refs]
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# python src\infer_image.py --config config\test\test_coderefformer_v5.yml --ckpt experiments\coderefformer_v5_5_refs_online_tokens\G_iter120000.pth --lq "D:\idpfvsr\datasets\CelebA-Test-Ref\celeba_test_split\test\lq\00228\02419.png" --ref_dir "D:\master\thesis\figures\experimente\02419_old"   

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeRefFormer inference — restore a face with HQ references.")
    parser.add_argument("--config",  required=True,
                        help="Training YML config path.")
    parser.add_argument("--ckpt",    required=True,
                        help="Generator checkpoint (.pth).")
    parser.add_argument("--lq",      required=True,
                        help="Path to the LQ input image.")
    parser.add_argument("--ref_dir", required=True,
                        help="Folder of HQ reference images.")
    parser.add_argument("--n_refs",  type=int, default=-1,
                        help="Number of references to use (-1 = all, default).")
    parser.add_argument("--output",  default="result.png",
                        help="Output image path (default: result.png).")
    parser.add_argument("--w",       type=float, default=1.0,
                        help="Fidelity weight w in [0,1] (default: 0.7).")
    parser.add_argument("--seed",    type=int,   default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load YML config
    with open(args.config) as f:
        opt = yaml.safe_load(f)

    # Override fidelity weight from command line
    opt['model']['generator']['w_scale'] = args.w

    # Point to the checkpoint
    opt['model']['g_load_path'] = args.ckpt

    # Disable discriminator (not needed at inference)
    opt['model']['d_load_path'] = None

    # Ensure we are in inference mode
    opt['is_train'] = False
    opt['device'] = 'cuda'

    # CodeFormerJointRefModel.__init__ reads opt['dataset']['train'] even at
    # inference time (ref_augment, latent_gt_path, etc.).
    # The training YML always has this block, but if it's missing for any reason
    # we add a minimal stub to avoid a KeyError.
    if 'dataset' not in opt:
        opt['dataset'] = {}
    if 'train' not in opt['dataset']:
        opt['dataset']['train'] = {}

    print(f"Building model from config: {args.config}")
    model = define_model(opt)
    model.net_G.eval()

    # Collect reference images (exclude HQ counterpart of the LQ input)
    lq_stem = Path(args.lq).stem
    ref_dir = Path(args.ref_dir)
    if ref_dir.is_file() and ref_dir.suffix.lower() in IMG_EXTENSIONS:
        ref_paths = [ref_dir]
        print(f"Using single reference image: {ref_dir}")
    else:
        ref_paths = collect_ref_paths(ref_dir, args.n_refs)
        excluded = [p for p in ref_paths if p.stem == lq_stem]
        ref_paths = [p for p in ref_paths if p.stem != lq_stem]
        if excluded:
            print(f"Excluded HQ counterpart: {[p.name for p in excluded]}")
        if not ref_paths:
            raise FileNotFoundError("No reference images left after excluding HQ counterpart.")
        print(f"Using {len(ref_paths)} reference image(s) from: {ref_dir}")
        for p in ref_paths:
            print(f"  {p.name}")

    # Load images
    img_size = 512   # CodeFormer operates at 512×512
    lq_tensor   = load_image_tensor(Path(args.lq),  img_size)         # (1, 3, 512, 512)
    ref_tensors = torch.cat([
        load_image_tensor(p, img_size) for p in ref_paths
    ], dim=0)                                                           # (N, 3, 512, 512)
    ref_tensors = ref_tensors.unsqueeze(0)                             # (1, N, 3, 512, 512)

    # Build data dict matching feed_data() expectations
    data = {
        'lr':  lq_tensor,      # (1, 3, H, W)
        'ref': ref_tensors,    # (1, N, 3, H, W)
    }

    # Run inference
    print(f"Running inference (w={args.w}) ...")
    result = model.infer(data)

    # Save output — replicate exactly what main.py does:
    #   post_process() clips [-1,1] -> [0,1] -> uint8, returns HWC RGB
    #   cv2.imwrite() needs BGR so we flip channels before writing
    sr_tensor   = result['hr_data']                          # (1, 3, H, W) RGB [-1, 1] on CPU
    sr_np       = data_utils.post_process(sr_tensor)         # (1, H, W, 3) RGB uint8
    sr_img      = sr_np[0]                                   # (H, W, 3) RGB uint8
    sr_bgr      = sr_img[..., ::-1]                          # RGB -> BGR for cv2
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sr_bgr.astype(np.uint8))
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()