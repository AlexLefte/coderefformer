"""
evaluate_predictions.py

Standalone evaluation script for saved predictions on disk.
Wraps the existing MetricCalculator without any dependency on the training loop.

Expected directory structure:

    Images are discovered recursively from pred_dir and gt_dir and matched
    by filename stem. The subfolder structure on either side is ignored.

        pred_dir/              gt_dir/
          img1.png      <->      img1.png
          subA/                  some_other_folder/
            img2.png    <->        img2.png
          subB/subC/
            img3.png    <->      img3.png

Usage examples:

    # Flat evaluation (individual images):
    python evaluate_predictions.py \
        --pred_dir results/my_model/predictions \
        --gt_dir data/test/gt \
        --metrics PSNR SSIM LPIPS IDS NIQE \
        --device cuda

    # Save results to JSON:
    python evaluate_predictions.py \
        --pred_dir results/my_model/predictions \
        --gt_dir data/test/gt \
        --metrics PSNR SSIM LPIPS IDS NIQE MUSIQ CLIPIQA \
        --save_path results/metrics.json \
        --model_tag my_model_v1

    # No-reference evaluation (no GT needed) - NIQE / MUSIQ / CLIPIQA only:
    python evaluate_predictions.py \
        --pred_dir results/my_model/predictions \
        --metrics NIQE MUSIQ CLIPIQA \
        --device cuda
"""

# python src/evaluate.py --gt_dir D:\idpfvsr\datasets\CelebA-Test-Ref\celeba_test_split\test\hq  --pred_dir D:\idpfvsr\repos\idpfvsr\experiments\results\ref_ldm\predict_ref_ldm_3_ref\CelebA-Test-Ref   

import argparse
import json
import os
import os.path as osp
import sys
from tqdm import tqdm

from metrics.metric_calculator import MetricCalculator
from collections import OrderedDict

import cv2
import numpy as np
import torch


# ─── Helpers ─────────────────────────────────────────────────────────────────

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')


def retrieve_images(directory):
    """
    Recursively walk `directory` and return a sorted list of all image paths found.
    Images are matched by filename stem, so the subfolder structure is ignored.
    """
    files = []
    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if fname.lower().endswith(IMG_EXTENSIONS):
                files.append(osp.join(root, fname))
    return sorted(files)


def load_image_rgb(path):
    """Load an image from disk in BGR and convert to RGB uint8 HWC array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if img.ndim == 2:
        # Grayscale -> replicate across 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = img[..., ::-1]  # BGR -> RGB
    return np.ascontiguousarray(img)


# ─── No-reference metrics (do not require a GT image) ────────────────────────

NO_REF_METRICS = {'NIQE', 'MUSIQ', 'CLIPIQA'}

# ─── Build the config dict expected by MetricCalculator ──────────────────────

def build_metric_opt(metrics, args):
    """
    Build the `opt['metric']` dictionary consumed by MetricCalculator.
    Default values for each metric can be overridden via argparse arguments.
    """
    metric_opt = OrderedDict()

    for m in metrics:
        m_upper = m.upper()

        if m_upper == 'PSNR':
            metric_opt['PSNR'] = {
                'colorspace': args.psnr_colorspace,
                'mult': 1,
            }
        elif m_upper == 'SSIM':
            metric_opt['SSIM'] = {'mult': 1}

        elif m_upper == 'LPIPS':
            metric_opt['LPIPS'] = {
                'model': 'net-lin',
                'net': 'alex',
                'colorspace': 'rgb',
                'spatial': False,
                'version': 0.1,
                'mult': 1,
            }
        elif m_upper == 'IDS':
            if not args.id_network_path:
                raise ValueError("--id_network_path is required for the IDS metric.")
            metric_opt['IDS'] = {
                'id_network_path': args.id_network_path,
                'mult': 1,
            }
        elif m_upper == 'IDSREF':
            if not args.id_network_path:
                raise ValueError("--id_network_path is required for the IDSRef metric.")
            metric_opt['IDSRef'] = {
                'id_network_path': args.id_network_path,
                'mult': 1,
            }
        elif m_upper == 'NIQE':
            metric_opt['NIQE'] = {'mult': 1}
        elif m_upper == 'MUSIQ':
            metric_opt['MUSIQ'] = {'mult': 1}
        elif m_upper == 'CLIPIQA':
            metric_opt['CLIPIQA'] = {'mult': 1}
        elif m_upper == 'TOF':
            metric_opt['tOF'] = {'mult': 1}
        else:
            raise ValueError(f"Unknown metric: {m}")

    return metric_opt


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(calculator, pred_dir, gt_dir, metrics):
    """
    Recursively collect all images from pred_dir and gt_dir, match them by
    filename stem (the name without extension), and evaluate in one pass.
    The folder structure on either side is completely ignored — only unique
    filenames matter.
    """
    pred_files = retrieve_images(pred_dir)

    need_gt = bool(set(metrics) - NO_REF_METRICS)

    if need_gt:
        gt_files = retrieve_images(gt_dir)

        # Build stem -> full path maps; stems must be unique across both trees
        pred_stems = {osp.splitext(osp.basename(f))[0]: f for f in pred_files}
        gt_stems   = {osp.splitext(osp.basename(f))[0]: f for f in gt_files}

        common = sorted(set(pred_stems) & set(gt_stems))
        if not common:
            raise RuntimeError(
                "No images with matching filenames found in pred_dir and gt_dir.\n"
                f"  pred_dir : {pred_dir}\n"
                f"  gt_dir   : {gt_dir}"
            )
        pred_paths = [pred_stems[s] for s in common]
        gt_paths   = [gt_stems[s]   for s in common]
        print(f"  Matched images: {len(common)}")
    else:
        # No-reference metrics only — GT is not needed
        pred_paths = pred_files
        gt_paths   = [None] * len(pred_files)
        print(f"  Images to evaluate (no-reference): {len(pred_paths)}")

    # Process one image at a time — load, evaluate, discard; no bulk accumulation
    for i, (pp, gp) in enumerate(tqdm(zip(pred_paths, gt_paths), total=len(pred_paths))):
        pred_img = load_image_rgb(pp)
        gt_img   = load_image_rgb(gp) if gp is not None else None

        # Add B size
        pred_img = np.expand_dims(pred_img, axis=0)
        gt_img   = np.expand_dims(gt_img, axis=0)

        calculator.compute_sequence_metrics(
            seq=str(i),
            true_seq_dir=gt_dir,
            pred_seq_dir=pred_dir,
            true_seq=gt_img,
            pred_seq=pred_img,
        )


# ─── Display si save rezultate ───────────────────────────────────────────────

def display_results(calculator, metrics):
    """Print per-sequence results and the global average to stdout."""
    print("\n" + "=" * 60)
    print("RESULTS PER SEQUENCE")
    print("=" * 60)

    for seq, metric_dict_per_seq in calculator.metric_dict.items():
        print(f"\nSequence: {seq}")
        for metric_type in calculator.metric_opt.keys():
            values = metric_dict_per_seq.get(metric_type, [])
            if values:
                mult = getattr(calculator, f'{metric_type.lower()}_mult', 1)
                print(f"  {metric_type:10s}: {mult * np.mean(values):.6f}  (x{mult})")

    print("\n" + "=" * 60)
    print("GLOBAL AVERAGE")
    print("=" * 60)
    avg = calculator.get_averaged_results()
    for metric_type, val in avg.items():
        mult = getattr(calculator, f'{metric_type.lower()}_mult', 1)
        print(f"  {metric_type:10s}: {mult * val:.6f}  (x{mult})")

    return avg


def save_results(avg_results, save_path, model_tag, calculator):
    """\n    Write averaged results to a JSON file.\n    If the file already exists, the new entry is merged in (keyed by model_tag).\n    """
    if osp.exists(save_path):
        with open(save_path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {}

    if model_tag not in existing:
        existing[model_tag] = {}

    for metric_type, val in avg_results.items():
        mult = getattr(calculator, f'{metric_type.lower()}_mult', 1)
        existing[model_tag][metric_type] = f'{mult * val:.6f}'

    existing = OrderedDict(sorted(existing.items()))
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(existing, f, sort_keys=False, indent=4)

    print(f"\nResults saved to: {save_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate image quality metrics on saved model predictions.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Directories ──────────────────────────────────────────────────────────
    parser.add_argument(
        '--pred_dir', required=True,
        help="Directory containing model predictions (PNG/JPG images)."
    )
    parser.add_argument(
        '--gt_dir', default=None,
        help="Directory containing ground-truth images.\n"
             "Required for reference-based metrics (PSNR, SSIM, LPIPS, IDS).\n"
             "Can be omitted when running no-reference metrics only (NIQE/MUSIQ/CLIPIQA)."
    )

    # ── Metrics ─────────────────────────────────────────────────────────────
    parser.add_argument(
        '--metrics', nargs='+',
        default=['LPIPS', 'IDS', 'NIQE', 'CLIPIQA'],
        help="List of metrics to compute.\n"
             "Available: PSNR SSIM LPIPS IDS NIQE MUSIQ CLIPIQA tOF\n"
             "Default: PSNR SSIM LPIPS"
    )

    # ── Metric configuration ──────────────────────────────────────────────────
    parser.add_argument(
        '--psnr_colorspace', default='y', choices=['y', 'rgb'],
        help="Color space used for PSNR: 'y' (YCbCr luma) or 'rgb'. Default: y"
    )
    parser.add_argument(
        '--gt_bit_depth', type=int, default=8,
        help="Bit depth of the GT images (8 for standard uint8). Default: 8"
    )
    parser.add_argument(
        '--id_network_path', default='external_utils/webface_r50.onnx',
        help="Path to the ArcFace model weights. Required for IDS / IDSRef metrics."
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        '--save_path', default=None,
        help="JSON file where results are saved (optional).\n"
             "If the file already exists, new results are merged in."
    )
    parser.add_argument(
        '--model_tag', default='model',
        help="Key used to identify this run in the results JSON. Default: 'model'"
    )

    # ── Device ───────────────────────────────────────────────────────────────
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
        help="PyTorch device. Default: cuda if available, otherwise cpu."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Basic input validation ────────────────────────────────────────────────────
    if not osp.isdir(args.pred_dir):
        print(f"[ERROR] pred_dir does not exist: {args.pred_dir}")
        sys.exit(1)

    metrics_upper = [m.upper() for m in args.metrics]
    need_gt = bool(set(metrics_upper) - NO_REF_METRICS)

    if need_gt and args.gt_dir is None:
        print("[ERROR] --gt_dir is required for reference-based metrics: "
              f"{set(metrics_upper) - NO_REF_METRICS}")
        sys.exit(1)

    if args.gt_dir and not osp.isdir(args.gt_dir):
        print(f"[ERROR] gt_dir does not exist: {args.gt_dir}")
        sys.exit(1)

    # ── Print configuration summary ─────────────────────────────────────────────────
    print("=" * 60)
    print("EVALUATION CONFIGURATION")
    print("=" * 60)
    print(f"  pred_dir   : {args.pred_dir}")
    print(f"  gt_dir     : {args.gt_dir or 'N/A (no-reference)'}")
    print(f"  metrics    : {metrics_upper}")
    print(f"  device     : {args.device}")
    print(f"  bit_depth  : {args.gt_bit_depth}")
    print()

    # ── Instantiate MetricCalculator ────────────
    metric_opt = build_metric_opt(metrics_upper, args)
    opt = {
        'metric':       metric_opt,
        'device':       args.device,
        'gt_bit_depth': args.gt_bit_depth,
    }

    # Define metric calculator
    calculator = MetricCalculator(opt)

    # ── Run evaluation ────────────────────────────────────────────────────
    print("Starting evaluation...")
    evaluate(calculator, args.pred_dir, args.gt_dir, metrics_upper)

    # ── Display and optionally save results ─────────────────────────────────────
    avg = display_results(calculator, metrics_upper)

    if args.save_path:
        save_results(avg, args.save_path, args.model_tag, calculator)


if __name__ == '__main__':
    main()