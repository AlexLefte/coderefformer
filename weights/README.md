# Pretrained weights

This folder holds pretrained checkpoints used for training/inference. Weight files
are not versioned in this repository — download them separately and place them here
(or point the config's `*_load_path` fields at wherever you keep them).

Expected layout referenced by the configs in `config/`:
- `weights/CodeFormer/` — CodeFormer generator/discriminator checkpoints
- `weights/facelib/` — face detection/parsing models (used by inference pre-processing)
- `weights/vqgan_code512.pth`, `weights/vqgan_code1024.pth` — pretrained VQ-VAE
- `weights/latent_gt_code512.pth`, `weights/latent_gt_code1024.pth` — precomputed HQ latent codes
