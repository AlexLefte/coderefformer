# ===========================================================================
# 1.  PCD ALIGNMENT MODULE  (inspired by EDVR, simplified to 2 pyramid levels)
# ===========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torchvision.ops import deform_conv2d


class PCDAlignmentModule(nn.Module):
    """Pyramidal Cascaded Deformable alignment module.

    Inspired by EDVR (Wang et al., 2019).  Simplified to 2 pyramid levels
    (coarse → fine) instead of EDVR's 3, which is sufficient for the spatial
    variation between face reference images.

    For each reference feature map the module:
      1. Computes coarse offsets at a downsampled resolution (level L).
      2. Upsamples those offsets by 2× and refines them at the original
         resolution (level L+1).
      3. Applies the final deformable convolution to spatially align the
         reference feature map to the LQ feature map.

    All offset networks are conditioned on the concatenation of LQ and
    reference features so the alignment is driven by their mutual content.

    Multi-reference support: each of the N references is aligned independently
    by merging the batch and reference dimensions (B*N), then split back.

    Args:
        in_ch   (int): Number of feature channels C at the *fine* level.
        n_groups (int): Number of deformable convolution offset groups.
                        Defaults to 8 (same as EDVR).
    """

    def __init__(self, in_ch: int, n_groups: int = 8):
        super().__init__()
        self.in_ch    = in_ch
        self.n_groups = n_groups
        # Number of offset channels for a 3×3 kernel: 2 * k*k * groups
        n_offset_ch = 2 * 3 * 3 * n_groups
        # Number of mask channels for a 3×3 kernel: k*k * groups
        n_mask_ch   = 3 * 3 * n_groups

        # --- Coarse level (operates on features downsampled 2×) ---
        # Offset network: takes cat(lq_coarse, ref_coarse) → offsets
        self.offset_conv_coarse = nn.Sequential(
            nn.Conv2d(2 * in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_ch, n_offset_ch + n_mask_ch, 3, padding=1),
        )

        # --- Fine level (operates on full-resolution features) ---
        # Receives upsampled coarse offsets as prior → refines them
        self.offset_conv_fine = nn.Sequential(
            # Additional n_offset_ch input channels for the upsampled coarse offsets
            nn.Conv2d(2 * in_ch + n_offset_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_ch, n_offset_ch + n_mask_ch, 3, padding=1),
        )

        # Deformable conv at fine level (weight = identity-init conv)
        self.deform_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)

        # Projection that reduces the aligned feature dimension back to in_ch
        # (same as EDVR's cascaded refinement conv)
        self.feat_fusion = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Zero-initialise the last conv of each offset network so that
        training starts from identity mapping (no deformation)."""
        for module in [self.offset_conv_coarse, self.offset_conv_fine]:
            last_conv = module[-1]
            nn.init.zeros_(last_conv.weight)
            nn.init.zeros_(last_conv.bias)

    def _split_offset_mask(self, raw: Tensor):
        """Split raw offset+mask tensor into (offset, mask).

        Args:
            raw (Tensor): (B, 2*k*k*G + k*k*G, H, W)
        Returns:
            offset (Tensor): (B, 2*k*k*G, H, W)
            mask   (Tensor): (B, k*k*G, H, W) — sigmoid-activated modulation mask
        """
        n_offset = 2 * 3 * 3 * self.n_groups
        offset = raw[:, :n_offset]
        mask   = torch.sigmoid(raw[:, n_offset:])
        return offset, mask

    def align(self, lq_feat: Tensor, ref_feat: Tensor) -> Tensor:
        """Align one reference feature map to the LQ feature map.

        Args:
            lq_feat  (Tensor): (B, C, H, W) — LQ encoder features
            ref_feat (Tensor): (B, C, H, W) — reference encoder features

        Returns:
            Tensor: (B, C, H, W) — spatially aligned reference features
        """
        # --- Coarse level: downsample 2× ---
        lq_c  = F.avg_pool2d(lq_feat,  kernel_size=2, stride=2)
        ref_c = F.avg_pool2d(ref_feat, kernel_size=2, stride=2)

        raw_coarse             = self.offset_conv_coarse(torch.cat([lq_c, ref_c], dim=1))
        offset_coarse, _       = self._split_offset_mask(raw_coarse)

        # Upsample coarse offsets to fine resolution (×2) — scale values accordingly
        offset_up = F.interpolate(offset_coarse, scale_factor=2,
                                  mode='bilinear', align_corners=False) * 2.0

        # --- Fine level: refine upsampled offsets ---
        raw_fine          = self.offset_conv_fine(
            torch.cat([lq_feat, ref_feat, offset_up], dim=1))
        offset_fine, mask = self._split_offset_mask(raw_fine)

        # Final deformable convolution using torchvision's deform_conv2d
        weight  = self.deform_conv.weight
        bias    = self.deform_conv.bias
        aligned = deform_conv2d(
            input         = ref_feat,
            offset        = offset_fine,
            weight        = weight,
            bias          = bias,
            padding       = 1,
            mask          = mask,
        )

        return self.feat_fusion(aligned)

    def forward(self, lq_feat: Tensor, ref_feats: Tensor) -> Tensor:
        """Align N reference feature maps to the LQ feature map.

        Args:
            lq_feat  (Tensor): (B, C, H, W)
            ref_feats (Tensor): (B, N, C, H, W)

        Returns:
            Tensor: (B, N, C, H, W) — each reference aligned to LQ space
        """
        B, N, C, H, W = ref_feats.shape

        # Merge batch and reference dims for parallel processing
        ref_flat = ref_feats.view(B * N, C, H, W)
        lq_exp   = lq_feat.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)

        aligned_flat = self.align(lq_exp, ref_flat)   # (B*N, C, H, W)

        return aligned_flat.view(B, N, C, H, W)