# CodeRefFormer

Reference-based face restoration built on top of CodeFormer's VQGAN codebook prior. A frozen VQ-VAE encodes the low-quality input, and a transformer predicts high-quality latent codes conditioned on features from a reference image/clip of the same identity. Includes image and video variants.

**Demo page (image and video comparisons):** [alexlefte.github.io/coderefformer](https://alexlefte.github.io/coderefformer/)

## Repository structure

```
src/                      Training, evaluation and inference code
  main.py                 Training entrypoint
  evaluate.py             Image model evaluation (metrics)
  evaluate_video.py       Video model evaluation (metrics)
  infer_image.py          Single-image inference
  infer_video.py          Video clip inference
  frames_to_mp4.py        Utility to assemble frame sequences into video

  data/                   Dataset loaders (paired / unpaired folder datasets)
  models/                 Model wrappers (loss computation, optimization, checkpointing)
    networks/             Generator/discriminator architectures
      codeformer_nets.py        Baseline CodeFormer (no reference)
      coderefformer_nets.py     CodeRefFormer (image, V1/V5)
      codeformer_video_nets.py  Video architectures (single-frame, ConvGRU, temporal-attention)
      vqgan_arch.py             VQ-VAE encoder/decoder + codebook
  metrics/                Metric computation (PSNR/SSIM/LPIPS/IDS/NIQE via pyiqa)
  utils/                  Shared data/training utilities

config/
  train/                  Training configs (image: CodeFormer baseline, CodeRefFormer V1/V3/V5)
  test/                   Testing/inference configs (image + video variants)

docs/                     Demo page (GitHub Pages, served from this folder)
```

## Model variants

| Config | Generator | Notes |
|---|---|---|
| `train_codeformer.yml` / `test_codeformer.yml` | `CodeFormer` | Baseline, no reference conditioning |
| `train_coderefformer_v1.yml` / `test_coderefformer_v1_good.yml` | `CodeRefFormer_V1` | Reference-conditioned via cross-attention |
| `train_coderefformer_v3.yml` / `test_coderefformer_v3_good.yml` | `CodeRefFormer_V1` (`use_hq_transformer: true`) | V1 with a frozen HQ reference transformer producing the cross-attention K/V |
| `train_coderefformer_v5.yml` / `test_coderefformer_v5.yml` | `CodeRefFormer_V5` | Latest image architecture |
| `test_video_codeformer_v0.yml` | `SingleFrameVideoVSR` | Video baseline, per-frame (no temporal modeling) |
| `test_video_coderefformer_v0.yml` | `SingleFrameVideoVSR` | Reference-conditioned, per-frame |
| `test_video_coderefformer_v1.yml` | `CodeFormerVSRv1` | ConvGRU-based temporal propagation |
| `test_video_coderefformer_v2.yml` | `TemporalAttnVideoVSR` | Temporal self-attention |

## Setup

```bash
pip install -r requirements.txt
```

`basicsr` is installed in editable mode directly from the CodeFormer repository (see `requirements.txt`) — it provides shared utilities (image I/O, degradations, arch utilities) reused here.

## Training

Run from inside `src/` (imports are resolved relative to this directory):

```bash
cd src
python main.py --exp_dir <experiment_dir> --mode train --opt ../config/train/train_coderefformer_v5.yml
```

`--mode` is one of `train`, `test`, `profile`. `--exp_dir` is where checkpoints/logs for the run are written.

## Inference

Run pretrained checkpoints on a single image or a video clip:

```bash
cd src
python infer_image.py --config ../config/test/test_coderefformer_v5.yml \
    --ckpt <path/to/G.pth> --lq <path/to/lq.png> --ref_dir <path/to/ref_images> --output result.png

python infer_video.py --config ../config/test/test_video_coderefformer_v2.yml \
    --ckpt <path/to/G.pth> --lq_dir <path/to/lq_frames> --ref_dir <path/to/ref_images> --out_dir output_frames
```

Run with `--help` on either script for the full set of options (`--n_refs`, `--w`, `--seed`, ...).

## Evaluation

`evaluate.py` / `evaluate_video.py` compute quality metrics (PSNR/SSIM/LPIPS/IDS/NIQE/CLIPIQA/...) over
already-generated predictions — run inference first, then point these at the output/GT directories:

```bash
cd src
python evaluate.py --pred_dir <predictions_dir> --gt_dir <gt_dir> --metrics LPIPS IDS NIQE CLIPIQA
python evaluate_video.py --pred_dir <predictions_root> --gt_dir <gt_root> --metrics LPIPS IDS EWarp CLIPIQA
```

`--pred_dir`/`--gt_dir` for the video script are root folders containing one sub-folder per sequence.
