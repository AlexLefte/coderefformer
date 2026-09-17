"""
evaluate_video.py

Standalone evaluation script for saved video predictions on disk.
Mirrors evaluate.py but handles sequences of frames grouped by folder.

Expected directory structure:

    pred_dir/              gt_dir/
      sequence_001/          sequence_001/
        00000.png    <->       00000.png
        00001.png    <->       00001.png
        ...                    ...
      sequence_002/          sequence_002/
        ...                    ...

Each top-level subfolder is treated as one video sequence.
Frames within a sequence are sorted alphabetically before processing,
so naming them with zero-padded indices (00000.png, 00001.png, ...) is
recommended to guarantee temporal order.

Usage examples:

    # Reference-based metrics:
    python evaluate_video.py \\
        --pred_dir results/my_model/test_heavy \\
        --gt_dir   data/test/gt \\
        --metrics  PSNR SSIM LPIPS IDS TOF EWARP \\
        --device   cuda

    # No-reference only (no GT needed):
    python evaluate_video.py \\
        --pred_dir results/my_model/test_heavy \\
        --metrics  NIQE MUSIQ CLIPIQA \\
        --device   cuda

    # Save results to JSON:
    python evaluate_video.py \\
        --pred_dir results/my_model/test_heavy \\
        --gt_dir   data/test/gt \\
        --metrics  PSNR SSIM LPIPS IDS TOF EWARP \\
        --save_path results/video_metrics.json \\
        --model_tag my_model_stage3
"""

import argparse
import json
import os
import os.path as osp
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from metrics.metric_calculator import MetricCalculator


# ─── Constants ───────────────────────────────────────────────────────────────

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# Metrics that do not require a ground-truth reference
NO_REF_METRICS = {'NIQE', 'MUSIQ', 'CLIPIQA'}


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def list_sequences(directory):
    """
    Recursively find all directories that directly contain image files.
    Returns a sorted list of paths relative to `directory`, so the structure
    can be arbitrarily deep (e.g. id/videos/video_id/*.png).
    """
    sequences = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        if any(f.lower().endswith(IMG_EXTENSIONS) for f in files):
            sequences.append(osp.relpath(root, directory))
    return sorted(sequences)


def list_frames(sequence_dir):
    """
    Return a sorted list of image file paths inside `sequence_dir`.
    Sorting is alphabetical, so zero-padded filenames give temporal order.
    """
    frames = sorted([
        osp.join(sequence_dir, f)
        for f in os.listdir(sequence_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    ])
    return frames


def load_frame_rgb(path):
    """Load a single frame from disk (BGR on disk → RGB uint8 HWC array)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load frame: {path}")
    if img.ndim == 2:
        # Grayscale → replicate to 3-channel RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = img[..., ::-1]  # BGR → RGB
    return np.ascontiguousarray(img)


def load_sequence(frame_paths):
    """
    Load all frames from a list of paths into a (T, H, W, C) uint8 numpy array.
    """
    frames = [load_frame_rgb(p) for p in frame_paths]
    return np.stack(frames, axis=0)  # (T, H, W, C)


# ─── Metric configuration ─────────────────────────────────────────────────────

def build_metric_opt(metrics, args):
    """
    Build the opt['metric'] dict consumed by MetricCalculator.
    Each entry maps a metric name to its configuration dict.
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
        elif m_upper == 'SIGMA_IDS':
            if not args.id_network_path:
                raise ValueError("--id_network_path is required for the Sigma_IDS metric.")
            metric_opt['Sigma_IDS'] = {
                'id_network_path': args.id_network_path,
                'mult': 1,
            }
        elif m_upper == 'NIQE':
            metric_opt['NIQE'] = {'mult': 1}
        elif m_upper == 'MUSIQ':
            metric_opt['MUSIQ'] = {'mult': 1}
        elif m_upper == 'CLIPIQA':
            metric_opt['CLIPIQA'] = {'mult': 1}
        elif m_upper in ('TOF', 'TOF'):
            # tOF: temporal optical flow consistency (video-only)
            metric_opt['tOF'] = {'mult': 1}
        elif m_upper == 'EWARP':
            # EWarp: warping error measuring temporal consistency
            metric_opt['EWarp'] = {'mult': 1}
        else:
            raise ValueError(f"Unknown metric: {m}")

    return metric_opt


# ─── Core evaluation loop ─────────────────────────────────────────────────────

def evaluate_video(calculator, pred_dir, gt_dir, metrics):
    """
    Iterate over all video sequences found in pred_dir.
    For each sequence, load all frames as a (T, H, W, C) array and pass it
    to MetricCalculator.compute_sequence_metrics — the same path used
    during training in main.py.

    If gt_dir is None (no-reference metrics only), GT sequences are skipped.
    """
    need_gt = bool(set(m.upper() for m in metrics) - NO_REF_METRICS)

    # Discover sequences in pred_dir
    pred_sequences = list_sequences(pred_dir)
    if not pred_sequences:
        raise RuntimeError(
            f"No sub-folders (sequences) found in pred_dir: {pred_dir}\n"
            "Each video sequence must be stored as a separate sub-folder."
        )

    if need_gt:
        gt_sequences = list_sequences(gt_dir)
        # Keep only sequences present in both directories
        common = sorted(set(pred_sequences) & set(gt_sequences))
        if not common:
            raise RuntimeError(
                "No matching sequence folders found between pred_dir and gt_dir.\n"
                f"  pred_dir sequences : {pred_sequences[:5]} ...\n"
                f"  gt_dir   sequences : {gt_sequences[:5]} ..."
            )
        print(f"  Matched sequences: {len(common)}")
    else:
        # No GT needed — evaluate all predicted sequences
        common = pred_sequences
        print(f"  Sequences to evaluate (no-reference): {len(common)}")

    # Process each sequence
    for seq_name in tqdm(common, desc="Sequences"):
        pred_seq_dir = osp.join(pred_dir, seq_name)
        gt_seq_dir   = osp.join(gt_dir, seq_name) if need_gt else None

        # Load predicted frames as (T, H, W, C) array
        pred_frame_paths = list_frames(pred_seq_dir)
        if not pred_frame_paths:
            print(f"  [WARN] No frames found in {pred_seq_dir}, skipping.")
            continue

        pred_seq = load_sequence(pred_frame_paths)  # (T, H, W, C)

        if need_gt:
            # Match GT frames by filename to handle missing frames gracefully
            gt_frame_paths = list_frames(gt_seq_dir)
            pred_stems = {osp.splitext(osp.basename(p))[0]: p for p in pred_frame_paths}
            gt_stems   = {osp.splitext(osp.basename(p))[0]: p for p in gt_frame_paths}

            matched_stems = sorted(set(pred_stems) & set(gt_stems))
            if not matched_stems:
                print(f"  [WARN] No matching frames for sequence '{seq_name}', skipping.")
                continue

            if len(matched_stems) < len(pred_frame_paths):
                print(
                    f"  [WARN] Sequence '{seq_name}': "
                    f"{len(pred_frame_paths) - len(matched_stems)} pred frames have no GT match."
                )

            # Re-load only the matched frames in temporal order
            pred_seq = load_sequence([pred_stems[s] for s in matched_stems])
            gt_seq   = load_sequence([gt_stems[s]   for s in matched_stems])
            gt_seq   = gt_seq[np.newaxis]    # (T,H,W,C) -> (1,T,H,W,C) for MetricCalculator
        else:
            gt_seq = None

        # MetricCalculator expects (1, T, H, W, C) for video — add batch dim
        pred_seq = pred_seq[np.newaxis]

        # Delegate to MetricCalculator — same API as in main.py
        calculator.compute_sequence_metrics(
            seq=seq_name,
            true_seq_dir=gt_seq_dir or '',
            pred_seq_dir=pred_seq_dir,
            true_seq=gt_seq,
            pred_seq=pred_seq,
        )


# ─── Display and save ─────────────────────────────────────────────────────────

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
                print(f"  {metric_type:12s}: {mult * np.mean(values):.6f}  (x{mult})")

    print("\n" + "=" * 60)
    print("GLOBAL AVERAGE")
    print("=" * 60)
    avg = calculator.get_averaged_results()
    for metric_type, val in avg.items():
        mult = getattr(calculator, f'{metric_type.lower()}_mult', 1)
        print(f"  {metric_type:12s}: {mult * val:.6f}  (x{mult})")

    return avg


def save_results(avg_results, save_path, model_tag, calculator):
    """
    Write averaged results to a JSON file.
    If the file already exists, the new entry is merged in (keyed by model_tag).
    """
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

    # Keep entries sorted alphabetically by model tag for readability
    existing = OrderedDict(sorted(existing.items()))
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(existing, f, sort_keys=False, indent=4)

    print(f"\nResults saved to: {save_path}")


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate video quality metrics on saved frame predictions.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Directories
    parser.add_argument(
        '--pred_dir', required=True,
        help="Root directory of model predictions.\n"
             "Must contain one sub-folder per video sequence."
    )
    parser.add_argument(
        '--gt_dir', default=None,
        help="Root directory of ground-truth frames.\n"
             "Required for reference-based metrics (PSNR, SSIM, LPIPS, IDS, tOF, EWarp).\n"
             "Can be omitted when running no-reference metrics only (NIQE/MUSIQ/CLIPIQA)."
    )

    # Metrics — defaults match the yml config
    parser.add_argument(
        '--metrics', nargs='+',
        default=['LPIPS', 'IDS', 'Sigma_IDS', 'EWarp', 'CLIPIQA'],
        help="Metrics to compute.\n"
             "Available: PSNR SSIM LPIPS IDS IDSREF SIGMA_IDS NIQE MUSIQ CLIPIQA TOF EWARP\n"
             "Default: IDS EWARP CLIPIQA"
    )

    # Metric options
    parser.add_argument(
        '--psnr_colorspace', default='y', choices=['y', 'rgb'],
        help="Color space for PSNR: 'y' (YCbCr luma) or 'rgb'. Default: y"
    )
    parser.add_argument(
        '--gt_bit_depth', type=int, default=8,
        help="Bit depth of GT images (8 for standard uint8). Default: 8"
    )
    parser.add_argument(
        '--id_network_path', default='external_utils/webface_r50.onnx',
        help="Path to ArcFace model weights. Required for IDS / IDSRef / Sigma_IDS."
    )

    # Output
    parser.add_argument(
        '--save_path', default=None,
        help="JSON file where results are written (optional).\n"
             "If the file already exists, new results are merged in."
    )
    parser.add_argument(
        '--model_tag', default='model',
        help="Key used to identify this run in the results JSON. Default: 'model'"
    )

    # Device
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
        help="PyTorch device. Default: cuda if available, otherwise cpu."
    )

    return parser.parse_args()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate input directories
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

    # Print configuration summary
    print("=" * 60)
    print("VIDEO EVALUATION CONFIGURATION")
    print("=" * 60)
    print(f"  pred_dir   : {args.pred_dir}")
    print(f"  gt_dir     : {args.gt_dir or 'N/A (no-reference)'}")
    print(f"  metrics    : {metrics_upper}")
    print(f"  device     : {args.device}")
    print(f"  bit_depth  : {args.gt_bit_depth}")
    print()

    # Build MetricCalculator
    metric_opt = build_metric_opt(metrics_upper, args)
    opt = {
        'metric':       metric_opt,
        'device':       args.device,
        'gt_bit_depth': args.gt_bit_depth,
    }
    calculator = MetricCalculator(opt)

    # Run evaluation
    print("Starting video evaluation...")
    evaluate_video(calculator, args.pred_dir, args.gt_dir, metrics_upper)

    # Display and optionally save results
    avg = display_results(calculator, metrics_upper)

    if args.save_path:
        save_results(avg, args.save_path, args.model_tag, calculator)


if __name__ == '__main__':
    main()