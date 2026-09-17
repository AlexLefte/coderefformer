"""
coderefformer_vsr_nets.py

Video face super-resolution architectures built on top of CodeRefFormerV1 / V5.

Variants, selectable via config:

  ConvGRUVideoVSR       — fully recurrent, causal, O(1) VRAM at inference.
                          Propagates latent state forward using a ConvGRU cell
                          and (optionally) flow-warped HQ prior.

  TemporalAttnVideoVSR  — global self-attention across the temporal window
                          (TemporalSpatialAttn) in the decoder.

  SingleFrameVideoVSR   — per-frame baseline (no temporal propagation).

All share:
  - Identical interface:  forward(lq, w, detach_16, early_feat, **kwargs)
  - Temporal decoder fusion (temporal_type: 'temp_attn' or 'dw').
  - Plug-and-play backbone: pass any CodeRefFormerV1 / V5 instance.
  - SpyNet (frozen) for optical-flow spatial alignment.
  - fix_modules config list for freezing arbitrary sub-modules.

Architecture overview
─────────────────────
                     LQ clip (B, T, C, H, W)
                          │
                     lq_encoder                ← frozen backbone
                          │
                     z_obs  (B, T, emb, 16, 16)
                          │
          ┌───────────────┴───────────────┐
          │   Temporal propagation module  │  ← TRAINED
          │  (ConvGRU  or  Bidirectional)  │
          └───────────────┬───────────────┘
                          │
                     z_fused_t  (B, emb, 16, 16)
                          │
              Transformer → Quantize             ← frozen backbone
                          │
              Generator + SFT/ID-SFT             ← fusion trained, rest frozen
                          │
                     out_t  (B, C, H, W)

Config example
──────────────
  type: ConvGRUVideoVSR           # or TemporalAttnVideoVSR / SingleFrameVideoVSR
  backbone_type: CodeRefFormerV5  # CodeRefFormerV1 | CodeRefFormerV5
  backbone_path: /path/to/ckpt.pth
  flownet_path:  /path/to/gmflow.pth

  emb_dim:        256
  hidden_dim:     64          # ConvGRU hidden channels
  use_flow:       true        # warp prev_out via optical flow for z_prior

  cfa_list:  ['16', '32']     # scales at which CFA is applied (KEEP default)
  cfa_nhead: 4

  fix_modules: ['encoder', 'quantize', 'generator', 'ft_layers',
                'feat_emb', 'idx_pred_layer', 'position_emb',
                'fuse_convs_dict', 'hq_encoder', 'flownet']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from einops import rearrange
from typing import Dict, List, Optional, Tuple

from basicsr.archs.arch_util import flow_warp
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

from .optical_flow_nets.spy_net import SpyNet

from .vqgan_arch import Encoder, ResBlock
from .codeformer_nets import CodeFormer
from .coderefformer_nets import (
    CodeRefFormerV1,
    CodeRefFormerV5,
)
# Registry of supported backbone classes, keyed by the string used in config.
# CodeRefFormerV1 also covers the former V3 (reference transformer) via its
# use_hq_transformer flag in the backbone config.
_BACKBONE_REGISTRY: Dict[str, type] = {
    'CodeFormer': CodeFormer,
    'CodeRefFormerV1': CodeRefFormerV1,
    'CodeRefFormerV5': CodeRefFormerV5,
}

def _build_backbone(backbone_type: str, backbone_cfg: dict) -> nn.Module:
    """Instantiate a backbone from its type name and config dict.

    Args:
        backbone_type: Key in _BACKBONE_REGISTRY (e.g. 'CodeRefFormerV1').
        backbone_cfg:  kwargs forwarded verbatim to the backbone constructor.
                       Keys not present in the constructor are absorbed by
                       **kwargs (all backbone classes accept **kwargs).
    Returns:
        Constructed backbone nn.Module.
    Raises:
        ValueError: if backbone_type is not in the registry.
    """
    logger = get_root_logger()
 
    if backbone_type not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone_type '{backbone_type}'. "
            f"Available: {list(_BACKBONE_REGISTRY.keys())}"
        )
 
    # Pop control keys so they are not forwarded to the constructor.
    cfg         = dict(backbone_cfg)          # shallow copy — don't mutate caller's dict
    ckpt_path   = cfg.pop('backbone_path', None)
    strict_load = cfg.pop('strict_load', False)

    cls = _BACKBONE_REGISTRY[backbone_type]
    backbone = cls(**cfg)
    logger.info(f'Backbone constructed: {backbone_type}')
 
    # Load checkpoint weights if a path was provided.
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # Support both raw state-dicts and checkpoint dicts with common keys.
        state_dict = (
            ckpt.get('params_ema') or
            ckpt.get('params')     or
            ckpt.get('state_dict') or
            ckpt
        )
        missing, unexpected = backbone.load_state_dict(state_dict, strict=strict_load)
        logger.info(f'Backbone loaded from: {ckpt_path} (strict={strict_load})')
        if missing:
            logger.warning(f'  Missing keys ({len(missing)}): {missing[:5]}{"..." if len(missing)>5 else ""}')
        if unexpected:
            logger.warning(f'  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
 
    return backbone


# ============================================================================
# 1.  Temporal decoder fusion blocks
# ============================================================================

class DepthwiseTemporalFusion(nn.Module):
    """Lightweight causal temporal fusion using depthwise separable convolution.

    Concatenates curr_feat and prev_feat along the channel dim, then applies:
      1. A pointwise conv (1×1) to mix the two frames per-channel → (B, C, H, W)
      2. A depthwise conv (3×3, groups=C) to aggregate local spatial context
         independently per channel — no cross-channel mixing, no global attention.
      3. A pointwise conv (1×1) to project back to C channels → delta
      4. Residual addition (curr_feat + delta). All output convs are zero-init,
         so delta starts at zero → identity at training start.

    Complexity: O(N × k²) with k=3, no quadratic term.
    At 256×256 with C=128: ~9 × 128 × 256² ≈ 75M multiply-adds — ~400× cheaper
    than full cross-attention at the same resolution.

    Rationale: consecutive frames differ only locally (small motion, lighting),
    so local 3×3 receptive field is sufficient to capture inter-frame deltas
    without the cost of global attention.

    Zero-init on all output weights → identity at training start, safe to add
    to a pre-trained backbone.

    Args:
        channels (int): Feature channels C.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Step 1: mix curr and prev channels → C  (pointwise)
        self.pw_in = nn.Conv2d(channels * 2, channels, kernel_size=1)
        nn.init.zeros_(self.pw_in.weight)
        nn.init.zeros_(self.pw_in.bias)

        # Step 2: local spatial aggregation per channel (depthwise 3×3)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3,
                            padding=1, groups=channels)
        nn.init.zeros_(self.dw.weight)
        nn.init.zeros_(self.dw.bias)

        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Step 3: project back to C  (pointwise)
        self.pw_out = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pw_out.weight)
        nn.init.zeros_(self.pw_out.bias)

    def forward(
        self,
        curr_feat: torch.Tensor,  # (B, C, H, W)
        prev_feat: torch.Tensor,  # (B, C, H, W)
    ) -> torch.Tensor:
        # Concatenate current and previous features along channel dim
        x = torch.cat([curr_feat, prev_feat], dim=1)  # (B, 2C, H, W)

        # Pointwise mix → depthwise local aggregation → pointwise project
        x = self.pw_in(x)       # (B, C, H, W)
        x = self.act(self.dw(x))
        delta = self.pw_out(x)  # (B, C, H, W)

        # Residual — delta is zero-init, so this is identity at training start
        return curr_feat + delta

# ============================================================================

# ============================================================================
# 2.  ConvGRU cell — spatial recurrent state in latent space
# ============================================================================

class ConvGRUCell(nn.Module):
    """Convolutional GRU cell operating on spatial feature maps (B, C, H, W).

    Standard GRU gating (reset, update, candidate) implemented as Conv2d
    so the hidden state preserves spatial structure.

    Args:
        input_dim  (int): Number of input channels.
        hidden_dim (int): Number of hidden-state channels.
        kernel_size (int): Convolution kernel size. Default 3.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        combined = input_dim + hidden_dim

        self.reset_gate  = nn.Conv2d(combined, hidden_dim, kernel_size, padding=pad)
        self.update_gate = nn.Conv2d(combined, hidden_dim, kernel_size, padding=pad)
        self.out_gate    = nn.Conv2d(combined, hidden_dim, kernel_size, padding=pad)

        self.hidden_dim  = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, H, W)
            h: (B, hidden_dim, H, W) or None  — initialised to zeros if None
        Returns:
            h_new: (B, hidden_dim, H, W)
        """
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_dim, x.shape[2], x.shape[3])

        combined = torch.cat([x, h], dim=1)

        r = torch.sigmoid(self.reset_gate(combined))   # reset gate
        z = torch.sigmoid(self.update_gate(combined))  # update gate

        combined_r = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.out_gate(combined_r))      # candidate state

        h_new = (1.0 - z) * h + z * n
        return h_new


# ============================================================================
# 3.  Temporal propagation modules
# ============================================================================

class LatentBlendNet(nn.Module):
    """Adaptive blending gate between z_obs and z_prior in latent space.

    Predicts a per-spatial-token alpha map from the GRU hidden state,
    then blends:  z_fused = alpha * z_prior + (1 - alpha) * z_obs

    Args:
        hidden_dim (int): ConvGRU hidden state channels.
        emb_dim    (int): Latent code channels (default 256 for VQGAN).
    """

    def __init__(self, hidden_dim: int, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )
        # Project hidden state to emb_dim for z_fused output
        self.proj = nn.Conv2d(hidden_dim, emb_dim, 1)

    def forward(
        self,
        h:       torch.Tensor,   # (B, hidden_dim, H, W)
        z_obs:   torch.Tensor,   # (B, emb_dim, H, W)
        z_prior: Optional[torch.Tensor],  # (B, emb_dim, H, W) or None
    ) -> torch.Tensor:
        """Returns z_fused (B, emb_dim, H, W)."""
        if z_prior is None:
            return z_obs

        alpha  = self.net(h)                              # (B, 1, 16, 16) in [0,1]
        return alpha * z_prior + (1.0 - alpha) * z_obs


class LatentResidualNet(nn.Module):
    """Residual correction blend between z_obs and temporal context from GRU.

    Instead of interpolating between z_obs and z_prior, computes a residual
    delta from the concatenation of z_obs, h, and optionally z_prior, then
    adds it back to z_obs with a learnable scalar gate:

        delta   = conv( cat(z_obs, h, [z_prior]) )
        z_fused = z_obs + tanh(gate) * delta

    Properties:
      - Zero-init on delta conv output → starts as identity (z_fused = z_obs)
      - gate is a learnable scalar, tanh-bounded to (-1, 1) → stable range
      - Works even when z_prior is None (first frame) — pads with zeros

    Args:
        hidden_dim (int): ConvGRU hidden state channels.
        emb_dim    (int): Latent code channels. Default 256.
        use_prior  (bool): Include z_prior as input. True when use_flow=True.
    """

    def __init__(self, hidden_dim: int, emb_dim: int = 256, use_prior: bool = True):
        super().__init__()
        self.use_prior = use_prior

        # Input channels: z_obs + h + (z_prior if use_prior)
        in_ch = emb_dim + hidden_dim + (emb_dim if use_prior else 0)

        self.delta_net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, emb_dim, 3, padding=1),
        )

        # Learnable scalar gate — zero-init → identity at start
        self.gate = nn.Parameter(torch.zeros(1))

        # Zero-init last conv → delta starts at zero
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

    def forward(
        self,
        h:       torch.Tensor,            # (B, hidden_dim, H, W)
        z_obs:   torch.Tensor,            # (B, emb_dim,    H, W)
        z_prior: Optional[torch.Tensor],  # (B, emb_dim,    H, W) or None
    ) -> torch.Tensor:
        """Returns z_fused (B, emb_dim, H, W)."""
        if self.use_prior:
            # Use z_prior if available, otherwise pad with zeros (first frame)
            prior = z_prior if z_prior is not None else torch.zeros_like(z_obs)
            inp   = torch.cat([z_obs, h, prior], dim=1)
        else:
            inp = torch.cat([z_obs, h], dim=1)

        delta   = self.delta_net(inp)              # (B, emb_dim, H, W)
        z_fused = z_obs + delta  # residual, identity at init
        return z_fused
    
def build_blend_net(
    blend_mode: str,
    hidden_dim: int,
    emb_dim:    int  = 256,
    use_prior:  bool = True,
) -> nn.Module:
    """Factory — returns LatentBlendNet or ResidualBlendNet based on blend_mode.

    Args:
        blend_mode: 'interpolate' for original LatentBlendNet,
                    'residual'    for new ResidualBlendNet.
        hidden_dim: ConvGRU hidden state channels.
        emb_dim:    Latent code channels.
        use_prior:  Whether z_prior is used (True when use_flow=True).
    """
    if blend_mode == 'interpolate':
        return LatentBlendNet(hidden_dim, emb_dim)
    elif blend_mode == 'residual':
        return LatentResidualNet(hidden_dim, emb_dim, use_prior=use_prior)
    else:
        raise ValueError(
            f"Unknown blend_mode '{blend_mode}'. "
            f"Expected 'interpolate' or 'residual'."
        )

# ============================================================================
# 4.  Base video VSR model — shared encoder/decoder logic
# ============================================================================

class BaseCodeFormerVSR(nn.Module):
    """Shared infrastructure for both VSR variants.

    Holds the backbone (CodeRefFormerV1 / V5), the frozen HQ encoder for
    flow-based prior computation, the frozen GMFlow network, and the CFA
    modules that are injected into the generator decoder loop.

    Subclasses implement `_propagate_latents(z_obs, lq_up)` which takes the
    full stack of LQ-encoded latent codes and returns `z_fused` per frame.

    Args:
        backbone         : Pre-built CodeRefFormerV1 or V5 instance.
        emb_dim    (int) : Latent code channels. Default 256.
        use_flow   (bool): Compute flow-warped HQ prior (z_prior). Default True.
        flownet_path (str|None): Path to GMFlow weights.
        cfa_list   (list): Generator decoder resolutions for CFA modules.
        cfa_nhead  (int) : Attention heads in CFA. Default 4.
        fix_modules (list): Sub-module names to freeze (requires_grad=False).
    """

    # Mapping: decoder output spatial size to feature channels (same as KEEP/CodeFormer)
    _CHANNELS = {'16': 512, '32': 256, '64': 256, '128': 128, '256': 128, '512': 64}

    # Generator block index after which each resolution is reached
    _FUSE_GEN_BLOCK = {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

    def __init__(
        self,
        backbone          = None,
        emb_dim:      int = 256,
        use_flow:     bool       = True,
        flownet_path: Optional[str] = None,
        temporal_list:     List[str]  = ['16', '32'],
        temporal_type: str = 'temp_attn',
        cfa_nhead:    int        = 4,
        fix_modules:  List[str]  = None,
        # Config-driven backbone construction — used when backbone is not
        # passed as a pre-built instance (standard case from YAML config).
        backbone_type: Optional[str]  = None,
        backbone_cfg:  Optional[dict] = None,
        hq_encoder_state_dict: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        logger = get_root_logger()

        # Build backbone from config if a pre-built instance was not provided.
        # backbone_type + backbone_cfg come directly from the generator section
        # of the YAML config; the subclass constructors forward them here via
        # **kwargs so the signature stays clean.
        if backbone is None:
            if backbone_type is None or backbone_cfg is None:
                raise ValueError(
                    "Either pass a pre-built 'backbone' instance or supply "
                    "both 'backbone_type' and 'backbone_cfg'."
                )
            backbone = _build_backbone(backbone_type, backbone_cfg)
            logger.info(f'Backbone built from config: {backbone_type}')

        self.backbone  = backbone
        self.emb_dim   = emb_dim
        self.use_flow  = use_flow

        # Frozen HQ encoder — re-encodes flow-warped prev_out to get z_prior.
        # Initialised from the backbone's own encoder weights (same VQGAN space).
        self.hq_encoder = Encoder(
            in_channels=3, nf=64, emb_dim=emb_dim,
            ch_mult=[1, 2, 2, 4, 4, 8], num_res_blocks=2,
            resolution=512, attn_resolutions=[16],
        )

        if hq_encoder_state_dict is not None:
            self.hq_encoder.load_state_dict(hq_encoder_state_dict, strict=True)
            logger.info('hq_encoder initialised from HQ VQGAN encoder weights.')
        else:
            # Fallback: warn loudly — backbone encoder lives in LQ latent space
            # and will produce incorrect z_prior values.  Always provide
            # hq_encoder_state_dict via CodeFormerVSRModel.set_network().
            logger.warning(
                'hq_encoder_state_dict not provided — falling back to '
                'backbone.encoder weights.  z_prior will be computed in LQ '
                'latent space, which is INCORRECT for HQ prev_out frames.  '
                'Pass hq_encoder_state_dict from the VQGAN checkpoint.'
            )
            self.hq_encoder.load_state_dict(
                backbone.encoder.state_dict(), strict=True)

        # Frozen optical flow network (GMFlow)
        if flownet_path is not None:
            self.flownet = SpyNet(load_path=flownet_path)
            self.flownet.eval()
        else:
            self.flownet = None
        logger.info(f'SpyNet loaded from: {flownet_path}')

        # Temporal decoder fusion modules — type controlled by temporal_type config
        # 'temp_attn' → TemporalSpatialAttn (global self-attention across T, parallel)
        # 'dw'        → DepthwiseTemporalFusion (lightweight depthwise fusion)
        self.temporal_list = temporal_list
        self.temporal_type = temporal_type
        self.temporal_fusion = nn.ModuleDict()

        for f_size in temporal_list:
            ch = self._CHANNELS[f_size]
            if temporal_type == 'temp_attn':
                self.temporal_fusion[f_size] = TemporalSpatialAttn(
                    channels=ch, nhead=cfa_nhead,
                )
            elif temporal_type == 'dw':
                self.temporal_fusion[f_size] = DepthwiseTemporalFusion(
                    channels=ch,
                )
            else:
                raise ValueError(
                    f"Unknown temporal_type '{temporal_type}'. "
                    f"Expected 'temp_attn' or 'dw'."
                )
        if temporal_list:
            logger.info(f'Temporal fusion ({temporal_type}) at scales: {temporal_list}')

        # Apply fix_modules (freeze listed sub-modules)
        if fix_modules:
            for name in fix_modules:
                # Support dotted paths: 'backbone.encoder', 'backbone.ft_layers', ...
                parts  = name.split('.')
                module = self
                try:
                    for p in parts:
                        module = getattr(module, p)
                    for param in module.parameters():
                        param.requires_grad = False
                    logger.info(f'Frozen: {name}')
                except AttributeError:
                    logger.warning(f'fix_modules: "{name}" not found — skipped.')

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _compute_flows(self, lq_up: torch.Tensor) -> torch.Tensor:
        """Compute forward optical flows between consecutive LQ frames.

        Args:
            lq_up: (B, T, C, H, W) LQ frames upsampled to full resolution.
        Returns:
            flows: (B, T-1, 2, H, W)
        """
        B, T, C, H, W = lq_up.shape
        f_curr = rearrange(lq_up[:, :-1], 'b t c h w -> (b t) c h w')
        f_next = rearrange(lq_up[:, 1:],  'b t c h w -> (b t) c h w')
        flows  = self.flownet(f_next, f_curr)           # (B*(T-1), 2, H, W)
        return flows.view(B, T - 1, 2, H, W)

    def _encode_lq_sequence(self, lq_up: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Run the backbone LQ encoder on all frames simultaneously.

        Args:
            lq_up: (B, T, C, H, W) full-resolution LQ frames.
        Returns:
            z_obs:         (B, T, emb_dim, 16, 16)
            enc_feat_dict: dict f_size → (B, T, C, h, w)  encoder features
                           needed for CFT/SFT in the generator.
        """
        B, T, C, H, W = lq_up.shape
        x = rearrange(lq_up, 'b t c h w -> (b t) c h w')

        enc_feat_dict = {}
        out_list = [self.backbone.fuse_encoder_block[f]
                    for f in self.backbone.connect_list]

        for i, block in enumerate(self.backbone.encoder.blocks):
            x = block(x)
            if i in out_list:
                f_size = str(x.shape[-1])
                enc_feat_dict[f_size] = rearrange(
                    x, '(b t) c h w -> b t c h w', b=B, t=T).detach()

        z_obs = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return z_obs, enc_feat_dict

    def _warp_and_encode(
        self,
        prev_out: torch.Tensor,       # (B, C, H, W)  previous restored frame
        flow:     torch.Tensor=None,  # (B, 2, H, W)  flow t-1 → t
    ) -> torch.Tensor:
        """Warp prev_out with flow, re-encode through hq_encoder → z_prior.

        Args:
            prev_out: restored frame from time t-1, pixel space.
            flow:     forward optical flow from t-1 to t.
        Returns:
            z_prior: (B, emb_dim, 16, 16)
        """
        if flow is not None:
            prev_out  = flow_warp(prev_out, flow.permute(0, 2, 3, 1))
        z_prior = self.hq_encoder(prev_out)
        return z_prior

    def _run_backbone_transformer(self, z):
        """Run the backbone's code-prediction transformer on a latent map.

        Mirrors the transformer stage of CodeRefFormer*.forward (position +
        feature embedding → SA layers → logits head). Shared by all video
        decode paths.

        Args:
            z: (batch, C, 16, 16) latent (batch is B or B*T depending on caller).
        Returns:
            logits: (batch, HW, codebook_size).
        """
        n = z.shape[0]
        pos_emb   = self.backbone.position_emb.unsqueeze(1).repeat(1, n, 1)
        query_emb = self.backbone.feat_emb(z.flatten(2).permute(2, 0, 1))
        for layer in self.backbone.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        return self.backbone.idx_pred_layer(query_emb).permute(1, 0, 2)

    def _decode_frame(
        self,
        z_fused:      torch.Tensor,   # (B, emb_dim, 16, 16)
        enc_feats_t:  dict,           # f_size → (B, C, h, w)
        prev_feats:   dict,           # f_size → (B, C, h, w)  for CFA
        detach_16:    bool = True,
        backbone_kwargs: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Run transformer → quantise → generator for one frame.

        Args:
            z_fused:      fused latent code for this frame.
            enc_feats_t:  encoder features at time t (for CFT/SFT skip connections).
            prev_feats:   decoder features from previous frame (for CFA).
            detach_16:    detach quant_feat before generator (Stage I/II training).
            backbone_kwargs: extra kwargs forwarded to backbone (id_embs, farl_embs…).

        Returns:
            out_t:       (B, C, H, W)  restored frame.
            logit_t:     (B, HW, codebook_size)  for cross-entropy loss.
            curr_feats:  dict f_size → (B, C, h, w)  decoder features this frame.
        """
        backbone_kwargs = backbone_kwargs or {}
        B = z_fused.shape[0]

        # ---- Transformer ----
        logit_t = self._run_backbone_transformer(z_fused)  # (B, HW, n)

        # ---- Quantize ----
        soft_one_hot = F.softmax(logit_t, dim=2)
        _, top_idx   = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat   = self.backbone.quantize.get_codebook_feat(
            top_idx, shape=[B, 16, 16, 256])

        if detach_16:
            quant_feat = quant_feat.detach()

        # ---- Generator with CFT/SFT + CFA ----
        x = quant_feat

        fuse_list = [self.backbone.fuse_generator_block[f]
                     for f in self.backbone.connect_list]
        temporal_fuse_list = [self.backbone.fuse_generator_block[f]
                      for f in self.temporal_list]

        curr_feats = {}

        for j, block in enumerate(self.backbone.generator.blocks):
            x = block(x)
            f_size = str(x.shape[-1])

            # CFT / SFT fusion (backbone)
            if j in fuse_list and f_size in enc_feats_t:
                enc_f = enc_feats_t[f_size]

                # CodeRefFormerV5: pass identity context
                if hasattr(self.backbone, 'ctx_builders'):
                    ctx = None
                    id_emb = backbone_kwargs.get('id_emb', None)
                    farl_seq    = backbone_kwargs.get('farl_seq',    None)
                    if id_emb is not None and f_size in self.backbone.ctx_builders:
                        ctx = self.backbone.ctx_builders[f_size](id_emb, farl_seq)
                    x = self.backbone.fuse_convs_dict[f_size](
                        enc_f.detach(), x,
                        backbone_kwargs.get('w', 0),
                        ctx=ctx,
                    )
                else:
                    # CodeRefFormerV1 / plain CodeFormer SFT.
                    # Always call fuse_convs_dict — the LQ-encoder skip-connections
                    # are required regardless of w; w only scales the SFT residual
                    # (w=0 means no reference guidance, but the skip-conn still runs).
                    x_ref_val = backbone_kwargs.get('x_ref', None)
                    w         = backbone_kwargs.get('w', 0)
                    
                    # Resolve reference features for this resolution (may be None)
                    ref_feats = (
                        x_ref_val[f_size].detach()
                        if (x_ref_val is not None and f_size in x_ref_val)
                        else None
                    )
                    # Pass both keyword forms: Fuse_sft_block_multiref uses
                    # enc_ref_feats (plural), Fuse_sft_block_ref uses enc_ref_feat
                    # (singular). The unused kwarg is silently absorbed by **kwargs.
                    x = self.backbone.fuse_convs_dict[f_size](
                        enc_f.detach(), x, w,
                        enc_ref_feats=ref_feats
                    )

            # Temporal fusion — behaviour depends on temporal_type:
            # 'cfa' / 'dcn': frame-by-frame, needs prev_feats
            # 'temp_attn':   not used here — handled in _decode_sequence()
            if j in temporal_fuse_list and f_size in self.temporal_fusion:
                if f_size in prev_feats:
                    x = self.temporal_fusion[f_size](x, prev_feats[f_size])
                # Always cache for next frame — even at t=0 when fusion doesn't run
                curr_feats[f_size] = x

        return x, logit_t, curr_feats

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement this
    # ------------------------------------------------------------------

    def _propagate_latents(
        self,
        z_obs:   torch.Tensor,   # (B, T, emb_dim, 16, 16)
        lq_up:   torch.Tensor,   # (B, T, C, H, W)  needed for flow computation
        flows:   Optional[torch.Tensor],  # (B, T-1, 2, H, W) or None
    ) -> torch.Tensor:
        """Return z_fused (B, T, emb_dim, 16, 16) — implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        lr:           torch.Tensor,           # (B, T, C, H, W) in [-1, 1]
        w:            float = 0,
        detach_16:    bool  = True,
        early_feat:   bool  = False,
        id_embs:      Optional[torch.Tensor] = None,  # (B, N, 512)  V5
        farl_embs:    Optional[torch.Tensor] = None,  # (B, N, 197, 768)  V5
        x_ref:        Optional[dict]         = None,  # V1 reference features
        **kwargs,
    ):
        """
        Args:
            lq:        LQ video clip (B, T, C, H, W) in [-1, 1].
            w:         SFT/ID-injection weight (0 = blind mode).
            detach_16: Detach quant_feat before generator.
            early_feat: Return (outs, logits, lq_feat, gen_feat_dict) if True,
                        else return outs only.

        Returns (training, early_feat=True):
            outs:          (B, T, C, H, W)
            logits:        (B*T, HW, codebook_size)
            lq_feat:       (B*T, emb_dim, 16, 16)  first-frame LQ feat
            gen_feat_dict: dict f_size → (B, T, C, h, w)
        Returns (inference, early_feat=False):
            outs: (B, T, C, H, W)
        """
        B, T, C, H, W = lr.shape

        # Encode all LQ frames simultaneously
        z_obs, enc_feat_dict = self._encode_lq_sequence(lr)
        # Flatten all T frames so shape is (B*T, C, 16, 16) — required by
        # hq_feat_loss which compares against quant_feat_gt of shape (B*T, ...).
        lq_feat = rearrange(z_obs, 'b t c h w -> (b t) c h w')

        # Compute optical flows (used by both variants for alignment)
        flows = self._compute_flows(lr) if self.use_flow else None

        # Temporal propagation → z_fused per frame
        z_fused_seq = self._propagate_latents(z_obs, lr, flows)  # (B, T, emb, 16, 16)

        # --- Early exit: only transformer + quantize, skip generator entirely ---
        # Used in Stage I training to optimize codebook loss + cross-entropy only.
        # ConvGRU/blend_net gradients still flow through z_fused_seq → lq_feat.
        if early_feat:
            z_flat    = rearrange(z_fused_seq, 'b t c h w -> (b t) c h w')
            logits = self._run_backbone_transformer(z_flat)
            # Return None for outs — generator not run
            return None, logits, lq_feat, {}

        # Build backbone kwargs dict (reference features, identity embeddings)
        backbone_kwargs = {'w': w, 'x_ref': x_ref}

        if id_embs is not None and hasattr(self.backbone, 'id_pool'):
            first_res   = self.backbone.connect_list[0]
            # Attention-pool N references → single embedding
            enc_f0      = enc_feat_dict[first_res][:, 0, ...]   # (B, C, h, w)
            backbone_kwargs['id_emb'] = self.backbone.id_pool(id_embs, enc_f0)

        if farl_embs is not None and hasattr(self.backbone, 'use_farl') \
                and self.backbone.use_farl:
            backbone_kwargs['farl_seq'] = farl_embs.mean(dim=1)

        # Decode frame-by-frame with temporal block
        outs       = []
        all_logits = []
        gen_feat_dict = defaultdict(list)
        prev_feats    = {}

        for t in range(T):
            z_t       = z_fused_seq[:, t]                      # (B, emb, 16, 16)
            enc_t     = {k: v[:, t] for k, v in enc_feat_dict.items()}

            out_t, logit_t, curr_feats = self._decode_frame(
                z_t, enc_t, prev_feats,
                detach_16=detach_16,
                backbone_kwargs=backbone_kwargs,
            )

            outs.append(out_t)
            all_logits.append(logit_t)
            prev_feats = curr_feats

            for f_size, feat in curr_feats.items():
                gen_feat_dict[f_size].append(feat)

        outs       = torch.stack(outs, dim=1)           # (B, T, C, H, W)
        all_logits = torch.stack(all_logits, dim=1)     # (B, T, HW, n)
        all_logits = rearrange(all_logits, 'b t l n -> (b t) l n')

        # Stack gen_feat_dict along temporal dim
        gen_feat_dict = {k: torch.stack(v, dim=1) for k, v in gen_feat_dict.items()}

        if self.training:
            return outs, all_logits, lq_feat, gen_feat_dict
        return outs


# ============================================================================
# 5.  ConvGRU variant — fully recurrent forward propagation
# ============================================================================

@ARCH_REGISTRY.register()
class ConvGRUVideoVSR(BaseCodeFormerVSR):
    """Recurrent video VSR using a ConvGRU cell in latent space.

    At each timestep:
      1. z_obs_t  = lq_encoder(LQ_t)
      2. z_prior_t = hq_encoder(flow_warp(prev_out, flow_{t-1→t}))   [if use_flow]
      3. h_t       = ConvGRU(cat(z_obs_t, z_prior_t), h_{t-1})
      4. z_fused_t = LatentBlendNet(h_t, z_obs_t, z_prior_t)
      5. out_t     = Decoder(Transformer(z_fused_t))

    The GRU cell provides temporal memory without requiring full-sequence
    pre-computation.  Causal: only past frames influence the current frame.

    Additional args vs BaseVideoVSR:
        hidden_dim (int): ConvGRU hidden-state channels. Default 64.
    """

    def __init__(
        self,
        backbone          = None,
        blend_mode:   str = 'residual',
        emb_dim:      int = 256,
        hidden_dim:   int = 64,
        use_flow:     bool = True,
        flownet_path: Optional[str] = None,
        temporal_list:     List[str] = ['16', '32'],
        temporal_type: str = 'dcn',
        cfa_nhead:    int = 4,
        fix_modules:  List[str] = None,
        backbone_type: Optional[str]  = None,
        backbone_cfg:  Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            blend_mode=blend_mode,
            emb_dim=emb_dim,
            use_flow=use_flow,
            flownet_path=flownet_path,
            temporal_list=temporal_list,
            temporal_type=temporal_type,
            cfa_nhead=cfa_nhead,
            fix_modules=fix_modules,
            backbone_type=backbone_type,
            backbone_cfg=backbone_cfg,
        )
        self.hidden_dim = hidden_dim

        # ConvGRU input: concatenation of z_obs and z_prior (if use_flow)
        gru_input_dim = emb_dim * 2 if use_flow else emb_dim
        self.gru_cell  = ConvGRUCell(gru_input_dim, hidden_dim)
        self.blend_net = build_blend_net(
            blend_mode = blend_mode,
            hidden_dim = hidden_dim,
            emb_dim    = emb_dim,
            use_prior  = use_flow,
        )

        self.training = kwargs.get('train', True)

    def _propagate_latents(
        self,
        z_obs:  torch.Tensor,
        lq_up:  torch.Tensor,
        flows:  Optional[torch.Tensor],
    ) -> torch.Tensor:
        # ConvGRUVideoVSR uses an interleaved recurrent loop in forward() so that
        # each frame's z_prior can be computed from the previous decoded output.
        # _propagate_latents cannot do that (it has no access to decoded frames),
        # so it is intentionally not used here.  Raise to catch accidental calls.
        raise NotImplementedError(
            "ConvGRUVideoVSR decodes frames recurrently inside forward(); "
            "_propagate_latents is not used by this subclass."
        )

    def forward(self, lr, w=0, detach_16=True, early_feat=False,
        id_embs=None, farl_embs=None, x_ref=None,
        h_init=None, prev_out_init=None, prev_feats_init=None,
        **kwargs):
        """Override forward to feed prev_out back into ConvGRU alignment.

        Two execution paths controlled by early_feat:

        early_feat=True  (Stage I training):
            Runs encoder + GRU + transformer only — no generator.
            Returns (None, logits, lq_feat, {}).
            Enables fast codebook / cross-entropy loss optimisation without
            the cost of the full generator forward pass.
            Note: z_prior is always None in this path because prev_out
            (pixel-space) is unavailable without the decoder. GRU still
            receives temporal context via its hidden state h.

        early_feat=False (Stage II training + inference):
            Full recurrent loop: encoder → GRU → transformer → generator.
            prev_out feeds back into z_prior for the next frame.
            Returns (outs, logits, lq_feat, gen_feat_dict) during training,
            or just outs during inference.
        """
        B, T, C, H, W = lr.shape
        z_obs, enc_feat_dict = self._encode_lq_sequence(lr)
        lq_feat = rearrange(z_obs, 'b t c h w -> (b t) c h w')
        flows = self._compute_flows(lr) if self.use_flow else None

        backbone_kwargs = {'w': w, 'x_ref': x_ref}
        if id_embs is not None and hasattr(self.backbone, 'id_pool'):
            first_res = self.backbone.connect_list[0]
            enc_f0    = enc_feat_dict[first_res][:, 0, ...]
            backbone_kwargs['id_emb'] = self.backbone.id_pool(id_embs, enc_f0)
        if farl_embs is not None and getattr(self.backbone, 'use_farl', False):
            backbone_kwargs['farl_seq'] = farl_embs.mean(dim=1)

        if early_feat:
            all_logits = []
            h          = None

            for t in range(T):
                z_obs_t = z_obs[:, t]

                # z_prior unavailable without pixel-space prev_out — pad with zeros.
                # GRU still accumulates temporal context via hidden state h.
                if self.use_flow:
                    gru_in = torch.cat([z_obs_t, torch.zeros_like(z_obs_t)], dim=1)
                else:
                    gru_in = z_obs_t

                h = self.gru_cell(gru_in, h)
                z_fused_t = self.blend_net(h, z_obs_t, z_prior=None)

                # Transformer only — skip quantize + generator
                logit_t = self._run_backbone_transformer(z_fused_t)
                all_logits.append(logit_t)

            all_logits = torch.stack(all_logits, dim=1)          # (B, T, HW, n)
            all_logits = rearrange(all_logits, 'b t l n -> (b t) l n')
            return None, all_logits, lq_feat, {}

        # ----------------------------------------------------------------
        # Stage II — full recurrent forward: encoder + GRU + decoder
        # ----------------------------------------------------------------
        outs          = []
        all_logits    = []
        gen_feat_dict = defaultdict(list)
        h        = h_init       
        prev_out = prev_out_init
        prev_feats = (
            {k: v for k, v in prev_feats_init.items()}
            if prev_feats_init is not None else {}
        )

        for t in range(T):
            z_obs_t = z_obs[:, t]

            z_prior_t = None
            if prev_out is not None:
                flow_t    = flows[:, t - 1] if (self.use_flow and t > 0) else None
                # At t=0 with prev_out_init: use flow from external source is unavailable,
                # so warp without flow (identity warp = just encode prev_out).
                z_prior_t = self._warp_and_encode(prev_out.detach(), flow_t)

            if self.use_flow:
                pad    = torch.zeros_like(z_obs_t) if z_prior_t is None else z_prior_t
                gru_in = torch.cat([z_obs_t, pad], dim=1)
            else:
                gru_in = z_obs_t

            h         = self.gru_cell(gru_in, h)
            z_fused_t = self.blend_net(h, z_obs_t, z_prior_t)

            enc_t = {k: v[:, t] for k, v in enc_feat_dict.items()}
            out_t, logit_t, curr_feats = self._decode_frame(
                z_fused_t, enc_t,
                prev_feats      = prev_feats,
                detach_16       = detach_16,
                backbone_kwargs = backbone_kwargs,
            )

            # Feed decoded output back for next frame's z_prior
            prev_out   = out_t
            prev_feats = curr_feats

            outs.append(out_t)
            all_logits.append(logit_t)
            for f, feat in curr_feats.items():
                gen_feat_dict[f].append(feat)

        outs = torch.stack(outs, dim=1)
        all_logits = torch.cat(all_logits, dim=0) 

        if self.training:
            # training path returns same tuple as before
            return outs, all_logits, lq_feat, gen_feat_dict

        # Inference: also return carry-out states
        return outs, h, prev_out, prev_feats


# ============================================================================
# 6.  Unidirectional variant — window-based forward + backward propagation
# ============================================================================

class TemporalSpatialAttn(nn.Module):
    """Global temporal self-attention applied independently per spatial position.
 
    For a feature tensor of shape (B, T, C, H, W), each spatial position
    (i, j) performs self-attention across the T time steps.  Spatial positions
    are kept independent (no cross-spatial mixing here — the existing
    Transformer handles that at the latent level).
 
    Complexity: O(T^2) per spatial position, O(T^2 * H * W) total.
    For T=5, H=W=16 this is 5^2 * 256 = 6400 attention pairs — negligible.
    For decoder resolutions (H=W=32/64) the same formula gives 5^2*1024=25600,
    still very manageable.
 
    The module is residual and zero-initialised at the output projection so it
    starts as an identity map and opens gradually during training.
 
    Args:
        channels (int): Feature channels C.
        nhead    (int): Number of attention heads. Must divide channels.
        dropout  (float): Attention dropout probability.
    """
 
    def __init__(self, channels: int, nhead: int = 4, dropout: float = 0.0):
        super().__init__()
        assert channels % nhead == 0, \
            f"channels ({channels}) must be divisible by nhead ({nhead})"
 
        self.norm   = nn.LayerNorm(channels)
        self.attn   = nn.MultiheadAttention(
            channels, nhead, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(channels)
        self.ff      = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
 
        # Zero-init output projection — starts as identity
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        # Zero-init FF output as well
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            x_out: (B, T, C, H, W) — temporally contextualised features
        """
        B, T, C, H, W = x.shape
 
        # Merge batch and spatial dims so each (b, i, j) position is independent
        # (B, T, C, H, W) → (B*H*W, T, C)
        x_in = rearrange(x, 'b t c h w -> (b h w) t c')
 
        # Self-attention across T
        x_norm = self.norm(x_in)                              # (B*H*W, T, C)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)      # (B*H*W, T, C)
        x_in = x_in + attn_out                               # residual
 
        # Feed-forward
        x_in = x_in + self.ff(self.norm_ff(x_in))            # residual
 
        # Reshape back to (B, T, C, H, W)
        return rearrange(x_in, '(b h w) t c -> b t c h w', b=B, h=H, w=W)

@ARCH_REGISTRY.register()
class TemporalAttnVideoVSR(BaseCodeFormerVSR):
    """Non-causal video VSR with global temporal self-attention at two levels.
 
    Replaces ConvGRU + CFA with TemporalSpatialAttn modules applied:
      - Once after the LQ encoder (latent level, 16x16 spatial)
      - After each SFT skip-connection in the generator decoder
        (decoder level, one module per resolution in connect_list)
 
    All T frames are processed in parallel throughout — no recurrence,
    no frame-by-frame loop.
 
    Additional args vs BaseCodeFormerVSR:
        nhead_latent        (int):       Attention heads at the latent level. Default 4.
        nhead_decoder       (int):       Attention heads at the decoder level. Default 4.
        temporal_list  (List[str]): Decoder resolutions where temporal attention
                                         is applied. Must be a subset of
                                         backbone.connect_list (e.g. ['32','64']).
                                         Defaults to all resolutions in connect_list.
    """
    def __init__(
        self,
        backbone                        = None,
        emb_dim:            int         = 256,
        nhead_latent:       int         = 4,
        # BaseCodeFormerVSR args
        use_flow:           bool        = False,
        flownet_path:       Optional[str] = None,
        temporal_list:      List[str]   = [],
        temporal_type:      str         = 'temp_attn',
        cfa_nhead:          int         = 4,
        fix_modules:        List[str]   = None,
        backbone_type:      Optional[str]  = None,
        backbone_cfg:       Optional[dict] = None,
        hq_encoder_state_dict: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            backbone              = backbone,
            emb_dim               = emb_dim,
            use_flow              = use_flow,
            flownet_path          = flownet_path,
            temporal_list         = temporal_list,
            temporal_type         = temporal_type,  # 'temp_attn' → TemporalSpatialAttn
            cfa_nhead             = cfa_nhead,
            fix_modules           = fix_modules,
            backbone_type         = backbone_type,
            backbone_cfg          = backbone_cfg,
            hq_encoder_state_dict = hq_encoder_state_dict,
        )

        # Latent-level temporal attention — specific to this architecture
        # Operates on z_obs (B, T, emb_dim, 16, 16) before the transformer
        self.latent_temporal_attn = TemporalSpatialAttn(
            channels = emb_dim,
            nhead    = nhead_latent,
        )
        # temporal_fusion (decoder-level) is built by BaseCodeFormerVSR
 
    # ------------------------------------------------------------------
    # _propagate_latents: global temporal attention on z_obs
    # ------------------------------------------------------------------
 
    def _propagate_latents(
        self,
        z_obs:  torch.Tensor,            # (B, T, emb, 16, 16)
        lq_up:  torch.Tensor,            # unused — kept for API compatibility
        flows:  Optional[torch.Tensor],  # unused
    ) -> torch.Tensor:
        """Apply global temporal self-attention over the full latent sequence.
 
        Args:
            z_obs: (B, T, emb, 16, 16)
        Returns:
            z_ctx: (B, T, emb, 16, 16) — temporally contextualised latents
        """
        return self.latent_temporal_attn(z_obs)
 
    # ------------------------------------------------------------------
    # _decode_sequence: parallel decode with decoder temporal attention
    # ------------------------------------------------------------------
 
    def _decode_sequence(
        self,
        z_ctx:           torch.Tensor,  # (B, T, emb, 16, 16)
        enc_feat_dict:   dict,          # f_size → (B, T, C, h, w)
        T:               int,
        detach_16:       bool = True,
        backbone_kwargs: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transformer → quantize → generator for all T frames in parallel.
 
        At each SFT skip-connection point:
          1. Apply frozen backbone SFT (encoder skip)
          2. Apply trained TemporalSpatialAttn across T frames
 
        Args:
            z_ctx:         (B, T, emb, 16, 16) — latent codes after temporal attn
            enc_feat_dict: f_size → (B, T, C, h, w) encoder features
            T:             number of frames
            detach_16:     detach quantised features before generator
            backbone_kwargs: w, x_ref, arcface_emb, farl_seq, ...
 
        Returns:
            outs:   (B, T, C, H, W)
            logits: (B*T, HW, codebook_size)
        """
        backbone_kwargs = backbone_kwargs or {}
        B = z_ctx.shape[0]
 
        # Flatten B and T for all per-frame operations
        z_flat = rearrange(z_ctx, 'b t c h w -> (b t) c h w')  # (B*T, emb, 16, 16)
 
        # ---- Transformer (per-frame, backbone frozen) ----
        logits = self._run_backbone_transformer(z_flat)
        # logits: (B*T, HW, codebook_size)
 
        # ---- Quantize (per-frame, backbone frozen) ----
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx   = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat   = self.backbone.quantize.get_codebook_feat(
            top_idx, shape=[B * T, 16, 16, 256])
 
        if detach_16:
            quant_feat = quant_feat.detach()
 
        # ---- Generator: all T frames in parallel (B*T batch) ----
        x = quant_feat   # (B*T, C, 16, 16)
 
        fuse_list = [self.backbone.fuse_generator_block[f]
                     for f in self.backbone.connect_list]
 
        for j, block in enumerate(self.backbone.generator.blocks):
            x = block(x)
            f_size = str(x.shape[-1])
 
            if j in fuse_list and f_size in enc_feat_dict:
                # Flatten encoder features from (B, T, C, h, w) to (B*T, C, h, w)
                enc_f = rearrange(
                    enc_feat_dict[f_size], 'b t c h w -> (b t) c h w')
 
                # --- Frozen backbone SFT skip-connection ---
                if hasattr(self.backbone, 'ctx_builders'):
                    # CodeRefFormerV5 path
                    ctx         = None
                    arcface_emb = backbone_kwargs.get('arcface_emb', None)
                    farl_seq    = backbone_kwargs.get('farl_seq',    None)
                    if arcface_emb is not None and f_size in self.backbone.ctx_builders:
                        ctx = self.backbone.ctx_builders[f_size](arcface_emb, farl_seq)
                    x = self.backbone.fuse_convs_dict[f_size](
                        enc_f.detach(), x,
                        backbone_kwargs.get('w', 0),
                        ctx=ctx,
                    )
                else:
                    # CodeRefFormerV1 path
                    w         = backbone_kwargs.get('w', 0)
                    x_ref_val = backbone_kwargs.get('x_ref', None)
                    ref_feats = None
                    if x_ref_val is not None and f_size in x_ref_val:
                        ref_f = x_ref_val[f_size]
                        # ref_f shape: (B, N, C, H, W) — N references per batch item
                        # fuse_convs_dict expects (B*T, N, C, H, W) — expand T dim
                        if ref_f.shape[0] == B:
                            # Expand from (B, N, C, H, W) → (B, T, N, C, H, W)
                            ref_f = ref_f.unsqueeze(1).expand(-1, T, -1, -1, -1, -1)
                            # Flatten B and T → (B*T, N, C, H, W)
                            ref_f = rearrange(ref_f, 'b t n c h w -> (b t) n c h w')
                        ref_feats = ref_f.detach()
                    x = self.backbone.fuse_convs_dict[f_size](
                        enc_f.detach(), x, w,
                        enc_ref_feats=ref_feats,
                    )
 
                # --- Trained decoder temporal attention ---
                # Reshape to (B, T, C, h, w), attend across T, reshape back
                if f_size in self.temporal_fusion:
                    x_seq = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
                    x_seq = self.temporal_fusion[f_size](x_seq)
                    x     = rearrange(x_seq, 'b t c h w -> (b t) c h w')
 
        # Reshape output to (B, T, C, H, W)
        outs = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        return outs, logits
 
    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        lr:           torch.Tensor,            # (B, T, C, H, W) in [-1, 1]
        w:            float = 0,
        detach_16:    bool  = True,
        early_feat:   bool  = False,
        id_embs:      Optional[torch.Tensor] = None,
        farl_embs:    Optional[torch.Tensor] = None,
        x_ref:        Optional[dict]         = None,
        **kwargs,
    ):
        """
        Two execution paths controlled by early_feat:

        early_feat=True  (Stage I training):
            Runs encoder + latent temporal attention + transformer only.
            No generator, no decoder temporal attention.
            Returns (None, logits, lq_feat, {}).

        early_feat=False (Stage II training + inference):
            Full forward: encoder → latent attn → transformer → generator
            + decoder temporal attention.
            Returns (outs, logits, lq_feat, {}) during training,
            or just outs during inference.
        """
        B, T, C, H, W = lr.shape

        # Encode all LQ frames simultaneously
        z_obs, enc_feat_dict = self._encode_lq_sequence(lr)
        lq_feat = rearrange(z_obs, 'b t c h w -> (b t) c h w')  # for codebook loss

        # Latent-level temporal attention — runs in both Stage I and II
        # Gradients flow through latent_temporal_attn in both stages
        z_ctx = self._propagate_latents(z_obs, lr, flows=None)   # (B, T, emb, 16, 16)

        # Build backbone kwargs
        backbone_kwargs = {'w': w, 'x_ref': x_ref}
        if id_embs is not None and hasattr(self.backbone, 'id_pool'):
            first_res = self.backbone.connect_list[0]
            enc_f0    = enc_feat_dict[first_res][:, 0, ...]
            backbone_kwargs['arcface_emb'] = self.backbone.id_pool(id_embs, enc_f0)
        if farl_embs is not None and getattr(self.backbone, 'use_farl', False):
            backbone_kwargs['farl_seq'] = farl_embs.mean(dim=1)

        # ------------------------------------------------------------------
        # Stage I — early exit: encoder + latent attn + transformer, no generator
        # ------------------------------------------------------------------
        if self.training and early_feat:
            # Flatten B and T for transformer
            z_flat    = rearrange(z_ctx, 'b t c h w -> (b t) c h w')
            logits = self._run_backbone_transformer(z_flat)
            # Generator and decoder temporal attention not run in Stage I
            return None, logits, lq_feat, {}

        # ------------------------------------------------------------------
        # Stage II — full forward: encoder + latent attn + generator + decoder attn
        # ------------------------------------------------------------------
        outs, logits = self._decode_sequence(
            z_ctx           = z_ctx,
            enc_feat_dict   = enc_feat_dict,
            T               = T,
            detach_16       = detach_16,
            backbone_kwargs = backbone_kwargs,
        )

        if self.training:
            return outs, logits, lq_feat, {}
        return outs


# ============================================================================
# 7.  Single-frame baseline — backbone only, no temporal modules
# ============================================================================

@ARCH_REGISTRY.register()
class SingleFrameVideoVSR(BaseCodeFormerVSR):
    """Single-frame baseline: runs the frozen CodeFormer backbone on each
    video frame independently, with no temporal modules (no GRU, no flow,
    no CFA).
 
    Useful for:
      - Evaluating the image-restoration backbone directly on video input.
      - Providing a fair upper-bound comparison against the temporal model.
      - Debugging — if temporal results are worse than this, the temporal
        modules are hurting rather than helping.
 
    The class accepts the same forward() signature as ConvGRUVideoVSR so it
    can be used as a drop-in replacement in any evaluation script.
 
    Config example:
        generator:
          backbone_type: CodeRefFormerV1
          backbone_cfg:
            ...
          # No emb_dim / hidden_dim / use_flow / flownet_path / cfa_list needed.
          # They are accepted via **kwargs and silently ignored.
    """
 
    def __init__(
        self,
        backbone          = None,
        emb_dim:      int = 256,
        backbone_type: Optional[str]  = None,
        backbone_cfg:  Optional[dict] = None,
        hq_encoder_state_dict: Optional[dict] = None,
        **kwargs,   # absorb use_flow, flownet_path, cfa_list, hidden_dim, etc.
    ):
        # Pass an empty cfa_list and no flownet — no temporal modules needed.
        # hq_encoder is still constructed (needed by BaseCodeFormerVSR) but
        # will never be called since _warp_and_encode is not used here.
        super().__init__(
            backbone               = backbone,
            emb_dim                = emb_dim,
            use_flow               = False,
            flownet_path           = None,
            temporal_list          = [],        # no CFA
            temporal_type          = '',
            fix_modules            = None,      # backbone already frozen via warmup
            backbone_type          = backbone_type,
            backbone_cfg           = backbone_cfg,
            hq_encoder_state_dict  = hq_encoder_state_dict,
        )
 
    def _propagate_latents(self, z_obs, lq_up, flows):
        # No temporal propagation — z_obs is used directly as z_fused.
        return z_obs
 
    def forward(
        self,
        lr:           torch.Tensor,
        w:            float = 0,
        detach_16:    bool  = True,
        early_feat:   bool  = False,
        id_embs:      Optional[torch.Tensor] = None,
        farl_embs:    Optional[torch.Tensor] = None,
        x_ref:        Optional[dict]         = None,
        **kwargs,
    ):
        """Process each frame independently through the backbone.
 
        Args:
            lr: (B, T, C, H, W) in [-1, 1]
        Returns (training, early_feat=True):
            outs    : (B, T, C, H, W)
            logits  : (B*T, HW, codebook_size)
            lq_feat : (B*T, emb_dim, 16, 16)
            gen_feat_dict: empty dict (no temporal features)
        Returns (inference, early_feat=False):
            outs: (B, T, C, H, W)
        """
        B, T, C, H, W = lr.shape
 
        # Encode all frames simultaneously
        z_obs, enc_feat_dict = self._encode_lq_sequence(lr)
        lq_feat = rearrange(z_obs, 'b t c h w -> (b t) c h w')
 
        backbone_kwargs = {'w': w, 'x_ref': x_ref}
        if id_embs is not None and hasattr(self.backbone, 'id_pool'):
            first_res = self.backbone.connect_list[0]
            enc_f0    = enc_feat_dict[first_res][:, 0, ...]
            backbone_kwargs['id_emb'] = self.backbone.id_pool(id_embs, enc_f0)
        if farl_embs is not None and getattr(self.backbone, 'use_farl', False):
            backbone_kwargs['farl_seq'] = farl_embs.mean(dim=1)
 
        # Decode each frame independently — no prev_feats (empty dict)
        outs, all_logits = [], []
        for t in range(T):
            z_t     = z_obs[:, t]                           # (B, emb, 16, 16)
            enc_t   = {k: v[:, t] for k, v in enc_feat_dict.items()}
            out_t, logit_t, _ = self._decode_frame(
                z_t, enc_t,
                prev_feats       = {},   # no CFA — no previous frame features
                detach_16        = detach_16,
                backbone_kwargs  = backbone_kwargs,
            )
            outs.append(out_t)
            all_logits.append(logit_t)
        outs = torch.stack(outs, dim=1)           # (B, T, C, H, W)
        
        if self.training and early_feat:
            all_logits = torch.stack(all_logits, dim=1)
            all_logits = rearrange(all_logits, 'b t l n -> (b t) l n')
            return outs, all_logits, lq_feat, {}
        # Return the same 4-tuple as ConvGRUVideoVSR at inference so the
        # chunked-inference path in codeformer_video_model can unpack uniformly.
        # State values are None because this model is stateless.
        return outs, None, None, None

# ============================================================================
# 7.  Bidirectional variant — window-based forward + backward propagation
# ============================================================================

class LatentPropagationBranch(nn.Module):
    """Single propagation branch (forward or backward) in latent space.

    At each step, aligns the hidden state from the previous step with the
    current frame using flow warp (optional), then fuses via residual blocks.

    Args:
        emb_dim    (int): Latent code channels.
        hidden_dim (int): Propagation branch channels.
        n_resblocks (int): Number of residual blocks for feature fusion.
    """

    def __init__(self, emb_dim: int = 256, hidden_dim: int = 128, n_resblocks: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project input + hidden state to hidden_dim
        self.input_proj = nn.Conv2d(emb_dim * 2, hidden_dim, 3, padding=1)

        # Residual fusion blocks
        self.resblocks = nn.Sequential(
            *[ResBlock(hidden_dim, hidden_dim) for _ in range(n_resblocks)]
        )

        # Project back to emb_dim
        self.out_proj = nn.Conv2d(hidden_dim, emb_dim, 3, padding=1)

        # Learnable residual gate — zero-init, starts silent
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z_obs:   torch.Tensor,              # (B, emb, 16, 16)
        hidden:  Optional[torch.Tensor],    # (B, emb, 16, 16) or None
        flow:    Optional[torch.Tensor],    # (B, 2, H, W) — flow to warp hidden
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_out:   (B, emb, 16, 16) — updated latent code
            hidden:  (B, emb, 16, 16) — new hidden state (= z_out)
        """
        if hidden is None:
            hidden = torch.zeros_like(z_obs)

        # Optionally warp hidden state to align with current frame
        if flow is not None:
            # Downsample flow to latent resolution (16x16)
            lat_h, lat_w = z_obs.shape[-2:]
            flow_down = F.interpolate(
                flow, size=(lat_h, lat_w), mode='bilinear', align_corners=False)
            hidden_warped = flow_warp(hidden, flow_down.permute(0, 2, 3, 1))
        else:
            hidden_warped = hidden

        x = self.input_proj(torch.cat([z_obs, hidden_warped], dim=1))
        x = self.resblocks(x)
        delta = self.gate.tanh() * self.out_proj(x)

        z_out = z_obs + delta
        return z_out, z_out

@ARCH_REGISTRY.register()
class BidirectionalVideoVSR(BaseCodeFormerVSR):
    """Bidirectional video VSR with forward + backward propagation in latent space.

    Mirrors BasicVSR's bidirectional hidden state propagation, but operates
    on VQGAN latent codes (16×16) instead of image feature maps.

    Pipeline (per temporal window):
      1.  Encode all T frames → z_obs  (B, T, emb, 16, 16)
      2.  Backward pass:  t = T-1 → 0, produce z_bwd  (B, T, emb, 16, 16)
      3.  Forward  pass:  t = 0 → T-1, produce z_fwd  (B, T, emb, 16, 16)
      4.  Fusion: z_fused = FusionNet(z_obs, z_fwd, z_bwd)
      5.  Decode each frame with CFT/SFT + CFA.

    Additional args vs BaseVideoVSR:
        hidden_dim  (int): Propagation branch channels. Default 128.
        n_resblocks (int): Residual blocks per branch. Default 2.
    """

    def __init__(
        self,
        backbone          = None,
        emb_dim:      int = 256,
        hidden_dim:   int = 128,
        n_resblocks:  int = 2,
        use_flow:     bool = True,
        flownet_path: Optional[str] = None,
        cfa_list:     List[str] = ['16', '32'],
        cfa_nhead:    int = 4,
        fix_modules:  List[str] = None,
        backbone_type: Optional[str]  = None,
        backbone_cfg:  Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            emb_dim=emb_dim,
            use_flow=use_flow,
            flownet_path=flownet_path,
            cfa_list=cfa_list,
            cfa_nhead=cfa_nhead,
            fix_modules=fix_modules,
            backbone_type=backbone_type,
            backbone_cfg=backbone_cfg,
        )

        self.fwd_branch = LatentPropagationBranch(emb_dim, hidden_dim, n_resblocks)
        self.bwd_branch = LatentPropagationBranch(emb_dim, hidden_dim, n_resblocks)

        # Fusion: combines z_obs + z_fwd + z_bwd
        self.fusion = nn.Sequential(
            nn.Conv2d(emb_dim * 3, emb_dim * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(emb_dim * 2, emb_dim, 3, padding=1),
        )

    def _propagate_latents(
        self,
        z_obs:  torch.Tensor,
        lq_up:  torch.Tensor,
        flows:  Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Bidirectional propagation in latent space.

        Args:
            z_obs:  (B, T, emb, 16, 16)
            lq_up:  (B, T, C, H, W)  — only used for flow (full res flows)
            flows:  (B, T-1, 2, H, W) forward flows t → t+1, or None
        Returns:
            z_fused: (B, T, emb, 16, 16)
        """
        B, T = z_obs.shape[:2]

        # ---- Backward pass: T-1 → 0 ----
        z_bwd = []
        hidden_bwd = None
        for t in reversed(range(T)):
            # Backward flow: flow from t to t-1 = reverse of forward flow t-1 → t
            flow_t = None
            if self.use_flow and flows is not None and t < T - 1:
                # flows[:, t] is flow from t to t+1
                # For backward: warp hidden from t+1 to t using flows[:, t] reversed
                # Approximation: use negative of forward flow
                flow_t = -flows[:, t]

            z_t, hidden_bwd = self.bwd_branch(z_obs[:, t], hidden_bwd, flow_t)
            z_bwd.append(z_t)

        z_bwd.reverse()
        z_bwd = torch.stack(z_bwd, dim=1)     # (B, T, emb, 16, 16)

        # ---- Forward pass: 0 → T-1 ----
        z_fwd = []
        hidden_fwd = None
        for t in range(T):
            flow_t = None
            if self.use_flow and flows is not None and t > 0:
                flow_t = flows[:, t - 1]      # flow from t-1 to t

            z_t, hidden_fwd = self.fwd_branch(z_obs[:, t], hidden_fwd, flow_t)
            z_fwd.append(z_t)

        z_fwd = torch.stack(z_fwd, dim=1)     # (B, T, emb, 16, 16)

        # ---- Fusion ----
        # Process all frames in parallel (no temporal dependency here)
        z_obs_flat  = rearrange(z_obs,  'b t c h w -> (b t) c h w')
        z_fwd_flat  = rearrange(z_fwd,  'b t c h w -> (b t) c h w')
        z_bwd_flat  = rearrange(z_bwd,  'b t c h w -> (b t) c h w')

        z_fused_flat = self.fusion(
            torch.cat([z_obs_flat, z_fwd_flat, z_bwd_flat], dim=1)
        )
        return rearrange(z_fused_flat, '(b t) c h w -> b t c h w', b=B, t=T)
