import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional, List
from .vqgan_arch import *
from.modules.pcd_module import PCDAlignmentModule

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# ===========================================================================
# 1.  Auxiliary Attention Blocks
# ===========================================================================

class PatchCrossAttention(nn.Module):
    """Patch-based spatial cross-attention between decoder and reference features.

    Complexity: O(N * k²) instead of O(N²) for full cross-attention, where
    N = H*W (spatial tokens) and k is the patch radius.

    Each query position (i,j) in the decoder attends only to reference
    tokens within a (2r+1)×(2r+1) window centred on (i,j).  This preserves
    spatial locality — the eye region in the decoder attends to the eye region
    of the reference, not the chin.

    Multi-reference: references are merged along the sequence dimension so
    the window for each query spans the same spatial neighbourhood across all
    N_ref references (total window size: (2r+1)² * N_ref tokens per query).

    Args:
        in_ch    (int): Feature channels C.
        nhead    (int): Number of attention heads. Default 4.
        radius   (int): Chebyshev neighbourhood radius. Default 3 → 7×7 window.
        dropout  (float): Attention dropout. Default 0.0.
    """

    def __init__(self, dec_ch: int, ref_ch: int, nhead: int = 4,
                 radius: int = 3, dropout: float = 0.0):
        super().__init__()
        self.in_ch  = dec_ch
        self.nhead  = nhead
        self.radius = radius
        self.scale  = (dec_ch // nhead) ** -0.5

        # Q from decoder (dec_ch), K/V from ref encoder (ref_ch)
        # both projected to dec_ch for attention
        self.q_proj   = nn.Conv2d(dec_ch, dec_ch, kernel_size=1)
        self.k_proj   = nn.Conv2d(ref_ch, dec_ch, kernel_size=1)
        self.v_proj   = nn.Conv2d(ref_ch, dec_ch, kernel_size=1)
        self.out_proj = nn.Conv2d(dec_ch, dec_ch, kernel_size=1)
        self.norm     = nn.GroupNorm(num_groups=min(32, dec_ch), num_channels=dec_ch)
        self.gate     = nn.Parameter(torch.zeros(1))
        self.dropout  = nn.Dropout(dropout)

    def _extract_patches(self, feat: Tensor, r: int) -> Tensor:
        """Extract local (2r+1)×(2r+1) patches for every spatial position.

        Args:
            feat (Tensor): (B, C, H, W)
            r    (int):    radius

        Returns:
            Tensor: (B, H, W, (2r+1)², C) — local neighbourhood per position
        """
        B, C, H, W = feat.shape
        k = 2 * r + 1
        # Pad so every position has a full window
        feat_pad = F.pad(feat, (r, r, r, r), mode='reflect')
        # unfold extracts sliding windows: (B, C*k*k, H*W)
        patches = feat_pad.unfold(2, k, 1).unfold(3, k, 1)
        # patches shape: (B, C, H, W, k, k)
        patches = patches.contiguous().view(B, C, H, W, k * k)
        patches = patches.permute(0, 2, 3, 4, 1)   # (B, H, W, k², C)
        return patches

    def forward(self, dec_feat: Tensor, ref_feats: Tensor) -> Tensor:
        """
        Args:
            dec_feat  (Tensor): (B, C, H, W) — current decoder feature map
            ref_feats (Tensor): (B, N, C, H, W) — reference encoder features

        Returns:
            Tensor: (B, C, H, W) — decoder features enriched by local ref attention
        """
        B, C, H, W = dec_feat.shape
        B, N, C_r, H_r, W_r = ref_feats.shape
        r = self.radius
        k = 2 * r + 1

        # Project queries from decoder
        q_map = self.q_proj(dec_feat)               # (B, C, H, W)

        # Aggregate reference features across N refs by averaging projections
        # (could also be learned but avg keeps complexity low)
        ref_flat = ref_feats.view(B * N, C_r, H_r, W_r)
        k_map    = self.k_proj(ref_flat).view(B, N, C, H, W).mean(dim=1)  # (B, C, H, W)
        v_map    = self.v_proj(ref_flat).view(B, N, C, H, W).mean(dim=1)  # (B, C, H, W)

        # Extract local patches for keys and values: (B, H, W, k², C)
        k_patches = self._extract_patches(k_map, r)
        v_patches = self._extract_patches(v_map, r)

        # Reshape for multi-head attention
        head_dim = C // self.nhead
        # q: (B*H*W, nhead, 1, head_dim)
        q = q_map.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
        q = q.view(B * H * W, 1, self.nhead, head_dim).permute(0, 2, 1, 3)

        # k, v: (B*H*W, nhead, k², head_dim)
        k_seq = k_patches.reshape(B * H * W, k * k, C)
        v_seq = v_patches.reshape(B * H * W, k * k, C)
        k_seq = k_seq.view(B * H * W, k * k, self.nhead, head_dim).permute(0, 2, 1, 3)
        v_seq = v_seq.view(B * H * W, k * k, self.nhead, head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention: (B*H*W, nhead, 1, k²)
        attn = torch.matmul(q, k_seq.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate values: (B*H*W, nhead, 1, head_dim)
        out = torch.matmul(attn, v_seq)
        out = out.permute(0, 2, 1, 3).reshape(B * H * W, C)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        out = self.out_proj(out)
        out = self.norm(out)

        # Gated residual
        return dec_feat + self.gate.tanh() * out

# ===========================================================================
# 2.  Transformer Blocks
# ===========================================================================

class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

# ===========================================================================
# 2.  SFT Blocks
# ===========================================================================

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        

    def forward(self, enc_feat, dec_feat, w=1, **kwargs):
        # Fusing encoder-decoder features
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))

        # Computing scale and shift
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


class Fuse_sft_block_ref(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        
        # Residual identity fusion blocks
        self.id_enc = ResBlock(2*in_ch, out_ch)

    def forward(self, enc_feat, dec_feat, w=1, enc_ref_feats=None):
        if enc_ref_feats is not None:
            fusion_input = torch.cat([enc_feat, enc_ref_feats], dim=1)
            enc_feat = self.id_enc(fusion_input)

        # Fusing encoder-decoder features
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))

        # Computing scale and shift
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


class MultiRefSEFusion(nn.Module):
    """Squeeze-and-Excite fusion for multiple reference feature maps.

    Computes a single global scalar weight per reference image, then returns
    a weighted sum of all reference feature maps.  The weights are conditioned
    on both the reference features themselves and the current decoder context
    (concatenation of LQ encoder features and decoder features).

    Args:
        in_ch (int):     Number of channels C in each reference feature map.
        reduction (int): Channel reduction factor for the bottleneck FC layer.

    Inputs:
        refs  (Tensor): HQ encoder features for all N references. Shape [B, N, C, H, W].
        query (Tensor): Current context = cat(enc_feat_lq, dec_feat).  Shape [B, 2C, H, W].

    Returns:
        Tensor: Weighted combination of reference features. Shape [B, C, H, W].
    """

    def __init__(self, in_ch: int, reduction: int = 4):
        super().__init__()
        # Global average pool — collapses spatial dims to a single descriptor per channel
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Two-layer MLP that maps the 3C pooled descriptor to a single relevance score
        self.fc = nn.Sequential(
            nn.Linear(in_ch * 3, max(in_ch // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_ch // reduction, 4), 1),
        )

    def forward(self, refs: Tensor, query: Tensor) -> Tensor:
        B, N, C, H, W = refs.shape

        # Broadcast query to match the N-reference dimension
        query_exp = query.unsqueeze(1).expand(B, N, -1, H, W)   # [B, N, 2C, H, W]

        # Concatenate each reference with the query context along the channel dim
        combined = torch.cat([refs, query_exp], dim=2)           # [B, N, 3C, H, W]

        # Global average pool over spatial dims, then flatten to a vector per reference
        gap_out = self.gap(combined.view(B * N, 3 * C, H, W)).view(B, N, 3 * C)

        # MLP produces one scalar per reference; softmax normalises across N refs
        weights = self.fc(gap_out).squeeze(-1).softmax(dim=1)    # [B, N]

        # Weighted sum across the N reference dimension
        return (refs * weights[:, :, None, None, None]).sum(dim=1)  # [B, C, H, W]


class MultiRefAttentionFusion(nn.Module):
    """Spatial cross-attention fusion for multiple reference feature maps.

    Unlike MultiRefSEFusion which assigns one global weight per reference,
    this module computes per-spatial-position weights — each pixel in the
    output can draw from a different reference image.

    The query is the concatenation of LQ encoder features and decoder features
    (2C channels), projected down to C before computing similarities.

    Args:
        in_ch (int): Number of channels C in each reference feature map.
                     The query is expected to have 2*in_ch channels.

    Inputs:
        refs  (Tensor): HQ encoder features for all N references. Shape [B, N, C, H, W].
        query (Tensor): Current context = cat(enc_feat_lq, dec_feat).  Shape [B, 2C, H, W].

    Returns:
        Tensor: Spatially-weighted combination of reference features. Shape [B, C, H, W].
    """

    def __init__(self, in_ch: int):
        super().__init__()
        # Project query from 2C → C to match the reference feature dimension
        self.query_proj = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1)
        # Per-reference key and value projections (1x1 conv, dimension-preserving)
        self.key_proj   = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.value_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def forward(self, refs: Tensor, query: Tensor) -> Tensor:
        B, N, C, H, W = refs.shape

        # Project and flatten the query to a sequence of HW position vectors
        q = self.query_proj(query).flatten(2)       # [B, C, HW]

        # Process all N references together by merging batch and ref dims
        refs_flat = refs.view(B * N, C, H, W)
        k = self.key_proj(refs_flat).flatten(2)     # [B*N, C, HW]
        v = self.value_proj(refs_flat).flatten(2)   # [B*N, C, HW]

        # Restore the reference dimension for the attention computation
        k = k.view(B, N, C, H * W)                 # [B, N, C, HW]
        v = v.view(B, N, C, H * W)                 # [B, N, C, HW]

        # Dot-product similarity between each output position and each reference,
        # scaled by sqrt(C).  Softmax is over N (not over HW), so each spatial
        # position independently selects how much to attend to each reference.
        attn = torch.einsum('bci,bnci->bni', q, k) / (C ** 0.5)  # [B, N, HW]
        attn = attn.softmax(dim=1)

        # Aggregate reference values weighted by the attention scores
        out = torch.einsum('bni,bnci->bci', attn, v)   # [B, C, HW]
        return out.view(B, C, H, W)


class Fuse_sft_block_multiref(nn.Module):
    """Spatial Feature Transform (SFT) block with multi-reference support.

    Extends the single-reference SFT block by first aggregating N reference
    feature maps [B, N, C, H, W] into a single map [B, C, H, W] via either
    SE or spatial-attention fusion, then applying the standard SFT modulation
    (scale + shift) to the decoder features.

    When no references are provided (enc_ref_feats=None) or fusion_type='none',
    the block degrades to the plain single-stream SFT block.

    Args:
        in_ch (int):        Number of input channels C.
        out_ch (int):       Number of output channels (usually equal to in_ch).
        fusion_type (str):  How to aggregate references:
                              'se'        — global scalar weight per reference (SE).
                              'attention' — per-pixel spatial attention across references.
                              'none'      — ignore references; plain SFT behaviour.
    """

    def __init__(self, in_ch: int, out_ch: int, fusion_type: str = 'attention'):
        super().__init__()
        assert fusion_type in ('se', 'attention', 'none'), \
            f"fusion_type must be 'se', 'attention', or 'none', got '{fusion_type}'"
        self.fusion_type = fusion_type

        # Reference aggregation module (only built when references are used)
        if fusion_type == 'se':
            self.ref_fusion = MultiRefSEFusion(in_ch)
        elif fusion_type == 'attention':
            self.ref_fusion = MultiRefAttentionFusion(in_ch)

        # Fuses the aggregated reference (C) with the LQ encoder feature (C) → C
        self.id_enc = ResBlock(2 * in_ch, out_ch)

        # Fuses the (optionally reference-enriched) encoder feature with the
        # decoder feature to produce the SFT modulation context → C
        self.encode_enc = ResBlock(2 * in_ch, out_ch)

        # Learnable affine parameters for the SFT modulation
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(
        self,
        enc_feat: Tensor,                           # LQ encoder features:   [B, C, H, W]
        dec_feat: Tensor,                           # Current decoder features: [B, C, H, W]
        w: float = 1.0,                             # SFT modulation strength (w=0 → skip)
        enc_ref_feats: Optional[Tensor] = None,     # HQ reference features: [B, N, C, H, W] or None
    ) -> Tensor:

        if enc_ref_feats is not None and self.fusion_type != 'none':
            # Build the query context by concatenating encoder and decoder features
            query   = torch.cat([enc_feat, dec_feat], dim=1)   # [B, 2C, H, W]
            # Aggregate N reference maps into a single descriptor
            ref_agg = self.ref_fusion(enc_ref_feats, query)    # [B, C, H, W]
            # Enrich the LQ encoder feature with reference information
            enc_feat = self.id_enc(torch.cat([enc_feat, ref_agg], dim=1))  # [B, C, H, W]

        # Standard SFT path: fuse (enriched) encoder feat with decoder feat,
        # then compute scale and shift for the affine modulation.
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale    = self.scale(enc_feat)
        shift    = self.shift(enc_feat)
        # Residual SFT modulation — w controls how strongly the reference guides output
        residual = w * (dec_feat * scale + shift)
        return dec_feat + residual

# ===========================================================================
# 3.  CodeRefFormer Architectures
# ===========================================================================

class CodeRefFormerV1(VQAutoEncoder):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9, 
                codebook_size=1024, latent_size=256,
                connect_list=['32', '64', '128', '256'],
                vqgan_path=None, **kwargs):
        super(CodeRefFormerV1, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest',2, [16], codebook_size)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'])

        # Freeze requested modules (e.g. quantize, generator) — used when only
        # the transformer / reference-conditioning path is trained.
        fix_modules = kwargs.get('fix_modules', None)
        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # --- LQ transformer (optionally reference-conditioned) ---
        # When use_hq_transformer is False (default), the LQ transformer is a
        # plain stack of self-attention layers.  When True, it uses SA+CA layers
        # whose cross-attention attends to K/V produced by a frozen HQ
        # transformer running on the reference features (the former V3 path).
        self.use_hq_transformer = kwargs.get('use_hq_transformer', False)
        adain_ref      = kwargs.get('adain_ref', False)
        adain_per_head = kwargs.get('adain_per_head', False)

        if self.use_hq_transformer:
            # LQ transformer — trainable, with cross-attention to HQ K/V
            self.ft_layers = nn.ModuleList([
                TransformerSACALayerV3(
                    embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp,
                    dropout=0.0, adain_ref=adain_ref, adain_per_head=adain_per_head)
                for _ in range(self.n_layers)])

            # HQ transformer — frozen KV extractor (same architecture as a plain
            # SA layer, initialised identically so it starts as a copy of ft_layers).
            self.hq_ft_layers = nn.ModuleList([
                TransformerSALayerKVExtractor(TransformerSALayer(
                    embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0))
                for _ in range(self.n_layers)])
            for p in self.hq_ft_layers.parameters():
                p.requires_grad = False
        else:
            # Standard SA transformer
            self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
                                        for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
        
        self.channels = {
            '16': 512, '32': 256, '64': 256,
            '128': 128, '256': 128, '512': 64,
        }
        self.gen_channels = {      # generator decoder channels at each resolution
            '16': 256, '32': 256, '64': 256,
            '128': 128, '256': 128, '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512':2, '256':5, '128':8, '64':11, '32':14, '16':18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16':6, '32': 9, '64':12, '128':15, '256':18, '512':21}

        # --- SFT fusion blocks ---
        self.use_deformable  = kwargs.get('use_deformable',  False)
        deform_n_groups = kwargs.get('deform_n_groups', 8)
        fusion_type     = kwargs.get('sft_fusion_type', '')
        self.multi_ref  = fusion_type in ('se', 'attention')
        print(f"[CodeRefFormerV1] multi_ref={self.multi_ref}, fusion_type={fusion_type}, "
              f"use_deformable={self.use_deformable} w/ {deform_n_groups} groups, "
              f"use_hq_transformer={self.use_hq_transformer} (adain_ref={adain_ref}, "
              f"adain_per_head={adain_per_head}).")

        # SFT Modules
        self.fuse_convs_dict = nn.ModuleDict()
        if self.use_deformable:
            self.pcd_align_dict = nn.ModuleDict()

        for f_size in self.connect_list:
            in_ch = self.channels[f_size]

            # PCD Alignment Module
            if self.use_deformable:
                self.pcd_align_dict[f_size] = PCDAlignmentModule(
                    in_ch, n_groups=deform_n_groups)

            # SFT block per se
            if self.multi_ref:
                self.fuse_convs_dict[f_size] = Fuse_sft_block_multiref(
                    in_ch, in_ch, fusion_type=fusion_type)
            else:
                self.fuse_convs_dict[f_size] = Fuse_sft_block_ref(in_ch, in_ch)

        # --- Patch cross-attention blocks (optional) ---
        use_patch_attn     = kwargs.get('use_patch_attn',     False)
        patch_attn_list    = kwargs.get('patch_attn_list',    connect_list)
        patch_attn_radius  = kwargs.get('patch_attn_radius',  3)
        patch_attn_nhead   = kwargs.get('patch_attn_nhead',   4)
        self.use_patch_attn = use_patch_attn
        self.attn_set       = set(patch_attn_list) if use_patch_attn else set()

        if use_patch_attn:
            self.patch_attn_dict = nn.ModuleDict()
            for f_size in patch_attn_list:
                dec_ch = self.gen_channels[f_size]   # generator decoder channels
                ref_ch = self.channels[f_size]        # HQ encoder channels
                self.patch_attn_dict[f_size] = PatchCrossAttention(
                    dec_ch = dec_ch,
                    ref_ch = ref_ch,
                    nhead  = patch_attn_nhead,
                    radius = patch_attn_radius,
                )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _run_hq_transformer(self, x_ref: Tensor):
        """Run the frozen HQ transformer and return per-layer K/V for cross-attention.

        x_ref: (B, N_ref, C, H, W), C=256, H=W=16.  The K/V of the N_ref
        references are stacked along the sequence dimension so the LQ
        transformer's cross-attention can attend to all of them jointly.
        """
        B, N_ref, C_feat, H, W = x_ref.shape
        S = H * W

        ref_flat = x_ref.view(B * N_ref, C_feat, H, W)

        pos = self.position_emb.unsqueeze(1).repeat(1, B * N_ref, 1)  # (S, B*N_ref, dim_embd)
        h   = self.feat_emb(ref_flat.flatten(2).permute(2, 0, 1))     # (S, B*N_ref, dim_embd)

        kv_per_layer = []
        with torch.no_grad():
            for layer in self.hq_ft_layers:
                h, k, v = layer(h, query_pos=pos)
                dim_embd = k.shape[-1]

                # (S, B*N_ref, dim_embd) → (S*N_ref, B, dim_embd)
                k = (k.view(S, B, N_ref, dim_embd)
                       .permute(0, 2, 1, 3)
                       .reshape(S * N_ref, B, dim_embd))
                v = (v.view(S, B, N_ref, dim_embd)
                       .permute(0, 2, 1, 3)
                       .reshape(S * N_ref, B, dim_embd))
                kv_per_layer.append((k, v))

        return kv_per_layer

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False,
                x_ref=None, **kwargs):

        # ################### Encoder #####################
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        lq_feat = x

        # ############# HQ K/V extraction (optional) ##############
        # Only when the reference-processing transformer is enabled and the
        # 16x16 reference features are available.
        kv_per_layer = None
        if self.use_hq_transformer and x_ref is not None and '16' in x_ref:
            kv_per_layer = self._run_hq_transformer(x_ref['16'])

        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1,x.shape[0],1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
        query_emb = feat_emb
        # Transformer encoder
        for i, layer in enumerate(self.ft_layers):
            if self.use_hq_transformer:
                ref_kv = kv_per_layer[i] if kv_per_layer is not None else None
                query_emb = layer(query_emb, ref_kv=ref_kv, query_pos=pos_emb)
            else:
                query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb) # (hw)bn
        logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only: # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx   = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat   = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])

        # Detach Latent Code 
        if detach_16:
            quant_feat = quant_feat.detach()

        # Perform AdaIN on Selected Code
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            f_size = str(x.shape[-1])

            # SFT fusion at connect_list resolutions
            if i in fuse_list and w > 0 and x_ref is not None and f_size in x_ref:
                ref_feats = x_ref[f_size].detach()

                # PCD alignment — align references to LQ encoder space before SFT
                if self.use_deformable and f_size in self.pcd_align_dict:
                    ref_feats = self.pcd_align_dict[f_size](
                        enc_feat_dict[f_size].detach(), ref_feats)

                if self.multi_ref:
                    x = self.fuse_convs_dict[f_size](
                        enc_feat_dict[f_size].detach(), x, w,
                        enc_ref_feats=ref_feats)
                else:
                    x = self.fuse_convs_dict[f_size](
                        enc_feat_dict[f_size].detach(), x, w,
                        enc_ref_feats=ref_feats[:, 0, ...])

                # Patch cross-attention at attn_set resolutions (after SFT if overlap)
                if (self.use_patch_attn and f_size in self.attn_set
                        and x_ref is not None and f_size in x_ref):
                    x = self.patch_attn_dict[f_size](x, x_ref[f_size].detach())

        return x, logits, lq_feat
    

# ===========================================================================
# 7.  CodeRefFormerV5 — identity injection via ArcFace + FaRL in the generator
# ===========================================================================
 
class IDEmbeddingAttentionPool(nn.Module):
    """Attention pooling over N ArcFace embeddings conditioned on LQ features.
 
    Computes a softmax-weighted sum of the N per-reference ArcFace embeddings
    where weights are driven by a global LQ descriptor — the reference whose
    identity is closest to the LQ image receives the highest weight.
 
    Args:
        id_dim   (int): ArcFace embedding dim. Default 512.
        lq_ch    (int): LQ feature channels for the GAP query.
        proj_dim (int): Internal projection dim. Default 256.
    """
 
    def __init__(self, id_dim: int = 512, lq_ch: int = 256, proj_dim: int = 256):
        super().__init__()
        self.q_proj = nn.Linear(lq_ch,  proj_dim)
        self.k_proj = nn.Linear(id_dim, proj_dim)
        self.scale  = proj_dim ** -0.5
 
    def forward(self, id_embs: Tensor, lq_feat: Tensor) -> Tensor:
        """
        Args:
            id_embs  (Tensor): (B, N, id_dim)
            lq_feat  (Tensor): (B, C, H, W)
        Returns:
            Tensor: (B, id_dim) — attention-weighted ArcFace embedding
        """
        lq_desc = lq_feat.mean(dim=[2, 3])                              # (B, C)
        q       = self.q_proj(lq_desc)                                  # (B, proj_dim)
        k       = self.k_proj(id_embs)                                  # (B, N, proj_dim)
        attn    = torch.bmm(q.unsqueeze(1), k.permute(0, 2, 1)) * self.scale
        attn    = torch.softmax(attn, dim=-1)                           # (B, 1, N)
        return torch.bmm(attn, id_embs).squeeze(1)                      # (B, id_dim)
 
 
class CompositeContextBuilder(nn.Module):
    """Builds a composite identity context sequence for generator cross-attention.
 
    Follows the paper "Reference-Guided Identity Preserving Face Restoration"
    (Zhou et al., 2025), Section 3.1 — Composite Context.
 
    Projects ArcFace and (optionally) FaRL embeddings into the feature space
    of the generator at a given resolution level (out_dim == C_level), then
    adds sinusoidal positional encoding as described in the paper (Eq. 1).
 
    Two modes:
 
      use_farl=False  (ArcFace only):
        ArcFace (B, 512) → Linear(512, C_level) → 1 token
        ctx shape: (1, B, C_level)
        No positional encoding needed for a single token.
 
      use_farl=True  (composite, as per paper):
        ArcFace (B, 512)      → Linear(512, C_level) →   1 token
        FaRL    (B, 197, 768) → Linear(768, C_level) → 197 tokens
        ctx shape: (198, B, C_level) + sinusoidal positional encoding.
        Encoding is FIXED (not learned), identical to the original Transformer
        paper (Vaswani et al., 2017).
 
    Multi-reference strategy
    ------------------------
    The paper trains with a single reference face. For multiple references,
    they average N forward passes (classifier-free guidance, Eq. 6). In our
    CodeFormer-based adaptation, we instead mean-pool the N reference
    embeddings BEFORE calling this builder (done in CodeRefFormerV5.forward):
        ArcFace: IDEmbeddingAttentionPool(N refs) → 1 vector  (B, 512)
        FaRL:    mean(N refs, dim=1)              → 1 sequence (B, 197, 768)
    This keeps the context at exactly 198 tokens regardless of N, so the
    sinusoidal encoding is always consistent with what was seen at training
    time. The encoding does NOT encode "which reference" — it only encodes
    the position within the fixed-length composite sequence (ArcFace=0,
    FaRL patches=1..197), which is the same meaning at training and inference.
 
    Args:
        out_dim     (int):  Generator channels at this level (C_level).
        arcface_dim (int):  ArcFace embedding dim. Default 512.
        farl_dim    (int):  FaRL token dim (ViT-B). Default 768.
        farl_tokens (int):  Number of FaRL output tokens. Default 197.
        use_farl    (bool): Include FaRL tokens. Default False.
    """
 
    def __init__(
        self,
        out_dim:     int,
        arcface_dim: int  = 512,
        farl_dim:    int  = 768,
        farl_tokens: int  = 197,
        use_farl:    bool = False,
    ):
        super().__init__()
        self.use_farl    = use_farl
        self.farl_tokens = farl_tokens
        self.out_dim     = out_dim
 
        # ArcFace: 1 projected token
        self.proj_arcface = nn.Linear(arcface_dim, out_dim)
 
        if use_farl:
            # FaRL: 197 projected tokens
            self.proj_farl = nn.Linear(farl_dim, out_dim)
 
            # Sinusoidal positional encoding — fixed buffer, not a learned parameter.
            # Shape: (ctx_len, 1, out_dim) so it broadcasts over the batch dim.
            # Using register_buffer so it moves with .to(device) and is saved in
            # state_dict (as a non-trainable entry), but has no gradient.
            ctx_len = 1 + farl_tokens   # 198
            self.register_buffer(
                'pos_emb',
                self._build_sinusoidal(ctx_len, out_dim),   # (198, 1, out_dim)
                persistent=True,
            )
 
        # Context sequence length exposed to cross-attention
        self.ctx_len = (1 + farl_tokens) if use_farl else 1
 
    @staticmethod
    def _build_sinusoidal(seq_len: int, d_model: int) -> Tensor:
        """Build a sinusoidal positional encoding table.
 
        Follows Vaswani et al. (2017) "Attention Is All You Need", Section 3.5.
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 
        Args:
            seq_len (int): Number of positions (198 for ArcFace+FaRL).
            d_model (int): Embedding dimension (C_level at each resolution).
 
        Returns:
            Tensor: (seq_len, 1, d_model) — the 1 is for batch broadcasting.
        """
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        # Divisor term: 10000^(2i/d_model) for i = 0, 1, ..., d_model//2-1
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )                                                                  # (d_model/2,)
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        return pe.unsqueeze(1)                          # (seq_len, 1, d_model)
 
    def forward(
        self,
        arcface_emb: Tensor,                    # (B, arcface_dim)
        farl_seq:    Optional[Tensor] = None,   # (B, 197, farl_dim) — required if use_farl
    ) -> Tensor:
        """Build the composite context sequence.
 
        Inputs are already pooled over N references by the caller:
            arcface_emb: single vector per sample  (B, 512)
            farl_seq:    single sequence per sample (B, 197, 768)
 
        Returns:
            Tensor: (ctx_len, B, C_level)
                ctx_len = 1    when use_farl=False
                ctx_len = 198  when use_farl=True
        """
        # ArcFace: (B, 512) → (B, C) → (1, B, C)
        arc_tok = self.proj_arcface(arcface_emb).unsqueeze(0)   # (1, B, C)
 
        if self.use_farl:
            assert farl_seq is not None, \
                "CompositeContextBuilder: farl_seq is required when use_farl=True"
            # FaRL: (B, 197, 768) → (B, 197, C) → (197, B, C)
            farl_tok = self.proj_farl(farl_seq).permute(1, 0, 2)  # (197, B, C)
 
            # Concatenate ArcFace token + FaRL tokens → (198, B, C)
            ctx = torch.cat([arc_tok, farl_tok], dim=0)
 
            # Add sinusoidal positional encoding — pos_emb is (198, 1, C),
            # broadcasts over B automatically.
            # Position 0   → ArcFace token   (global identity)
            # Positions 1-197 → FaRL patch tokens (spatial face regions)
            ctx = ctx + self.pos_emb
        else:
            ctx = arc_tok   # (1, B, C) — no positional encoding for a single token
 
        return ctx
 
class IDCrossAttentionSFT(nn.Module):
    """SFT block with identity cross-attention, following the FIR-Adapter design.
 
    Pipeline (per resolution level):
 
        1. encode_enc  — fuse LQ encoder + decoder features (identical to
                         Fuse_sft_block), producing a spatial feature map F.
 
        2. ID cross-attention — F is the query; ctx (ArcFace + FaRL tokens
                         projected by CompositeContextBuilder) are K and V.
                         Gated residual keeps the block silent at init.
 
        3. LayerNorm   — normalise the attended spatial map.
 
        4. Conv2d → G (scale), Conv2d → B (shift) — produce per-pixel
                         affine parameters from the identity-modulated map.
 
        5. SFT         — dec_feat = dec_feat + w * (dec_feat * G + B)
 
    This places identity injection *after* the encoder–decoder fusion so the
    cross-attention query already carries structural context, making it easier
    for the model to spatially route identity signals.
 
    Args:
        in_ch  (int): Feature channels C at this resolution (== C_level).
        out_ch (int): Output channels (usually == in_ch).
        nhead  (int): Attention heads for cross-attention. Default 4.
    """
 
    def __init__(self, in_ch: int, out_ch: int, nhead: int = 4):
        super().__init__()
 
        # Step 1 — fuse LQ encoder + decoder.
        # Named 'encode_enc' to match Fuse_sft_block — loaded automatically
        # from a CodeFormer checkpoint with strict=False.
        self.encode_enc = ResBlock(2 * in_ch, in_ch)
 
        # Step 2 — cross-attention: Q = fused spatial tokens, K/V = ctx.
        # New weights — not present in CodeFormer, trained from scratch.
        self.cross_attn = nn.MultiheadAttention(
            in_ch, nhead, dropout=0.0, batch_first=False)
        # Learnable gate — zero-init keeps CA silent at training start
        self.ca_gate = nn.Parameter(torch.zeros(1))
 
        # Step 3 — LayerNorm on the attended spatial tokens.
        # New weight — not present in CodeFormer, trained from scratch.
        self.norm = nn.LayerNorm(in_ch)
 
        # Steps 4+5 — scale and shift branches.
        # Named 'scale' / 'shift' and architecture (Conv3x3 → LeakyReLU → Conv3x3)
        # matches Fuse_sft_block exactly — loaded automatically with strict=False.
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
 
    def forward(
        self,
        enc_feat: Tensor,                  # (B, C, H, W) — LQ encoder features
        dec_feat: Tensor,                  # (B, C, H, W) — current decoder features
        w:        float = 1.0,
        ctx:      Optional[Tensor] = None, # (ctx_len, B, C) — identity context
        **kwargs,
    ) -> Tensor:
 
        B, C, H, W = dec_feat.shape
 
        # --- Step 1: fuse encoder + decoder ---
        fused = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))  # (B, C, H, W)
 
        # --- Step 2: ID cross-attention (only when ctx is available) ---
        if ctx is not None:
            # Flatten spatial dims → sequence: (HW, B, C)
            q = fused.flatten(2).permute(2, 0, 1)
 
            # Cross-attention with identity context as K and V
            attn_out, _ = self.cross_attn(q, ctx, ctx)          # (HW, B, C)
 
            # Gated residual — gate starts silent, opens gradually
            q = q + self.ca_gate.tanh() * attn_out
 
            # --- Step 3: LayerNorm ---
            q = self.norm(q)
 
            # Reshape back to spatial map
            fused = q.permute(1, 2, 0).view(B, C, H, W)         # (B, C, H, W)
 
        # --- Steps 4+5: Conv → scale/shift → SFT ---
        scale = self.scale(fused)                                # (B, C, H, W)
        shift = self.shift(fused)                                # (B, C, H, W)
        return dec_feat + w * (dec_feat * scale + shift)
 
class CodeRefFormerV5(VQAutoEncoder):
    """CodeRefFormer V5 — identity injection via composite context in the generator.
 
    At every resolution in connect_list, the decoder feature map is modulated
    by a composite identity context built from ArcFace embeddings and optionally
    FaRL ViT-B tokens, then by the standard LQ-encoder SFT.
 
    The transformer is identical to V1 (pure SA, no reference signal) —
    identity enters only in the generator, keeping the codebook prediction
    objective clean and conflict-free.
 
    Two modes controlled by use_farl:
 
      use_farl=False:
        Context = 1 ArcFace token per resolution level.
        Cross-attention is a global identity modulation.
 
      use_farl=True:
        Context = 1 ArcFace token + 197 FaRL tokens = 198 tokens.
        Cross-attention is spatially selective — decoder positions attend
        to corresponding facial regions in the reference (via FaRL patches).
 
    Constructor kwargs
    ------------------
    arcface_dim  (int):  ArcFace embedding dim. Default 512.
    use_farl     (bool): Include FaRL ViT-B tokens. Default False.
    farl_dim     (int):  FaRL token dim (ViT-B output). Default 768.
    farl_tokens  (int):  FaRL sequence length. Default 197.
    id_nhead     (int):  Attention heads in ID cross-attention. Default 4.
    sft_fusion_type (str): LQ-SFT aggregation. Default 'attention'.
    """
 
    def __init__(
        self,
        dim_embd:      int       = 512,
        n_head:        int       = 8,
        n_layers:      int       = 9,
        codebook_size: int       = 1024,
        latent_size:   int       = 256,
        connect_list:  List[str] = ['32', '64', '128', '256'],
        vqgan_path:    Optional[str] = None,
        **kwargs,
    ):
        super().__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], codebook_size)
 
        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'])
 
        self.connect_list = connect_list
        self.n_layers      = n_layers
        self.dim_embd      = dim_embd
        self.dim_mlp       = dim_embd * 2
 
        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb     = nn.Linear(256, self.dim_embd)
 
        # Standard SA transformer — identical to V1, no reference signal
        self.ft_layers = nn.Sequential(*[
            TransformerSALayer(embed_dim=dim_embd, nhead=n_head,
                               dim_mlp=self.dim_mlp, dropout=0.0)
            for _ in range(self.n_layers)
        ])
 
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False),
        )
 
        self.channels = {
            '16': 512, '32': 256, '64': 256,
            '128': 128, '256': 128, '512': 64,
        }
        self.fuse_encoder_block   = {'512': 2, '256': 5, '128': 8,
                                     '64': 11, '32': 14, '16': 18}
        self.fuse_generator_block = {'16': 6, '32': 9, '64': 12,
                                     '128': 15, '256': 18, '512': 21}
 
        # --- Context configuration ---
        arcface_dim  = kwargs.get('arcface_dim',     512)
        use_farl     = kwargs.get('use_farl',        False)
        farl_dim     = kwargs.get('farl_dim',        768)
        farl_tokens  = kwargs.get('farl_tokens',     197)
        id_nhead     = kwargs.get('id_nhead',        4)
        fusion_type  = kwargs.get('sft_fusion_type', 'attention')
        self.use_farl = use_farl
 
        # Attention pooling: N ArcFace embeddings → 1 per batch element.
        # Query conditioned on the first encoder resolution feature map.
        lq_ch_pool = self.channels[connect_list[0]]
        self.id_pool = IDEmbeddingAttentionPool(
            id_dim=arcface_dim, lq_ch=lq_ch_pool, proj_dim=256)
 
        # One CompositeContextBuilder + one IDCrossAttentionSFT per resolution.
        # Each builder projects to C_level (varies per resolution) — independent
        # projection weights per level, no shared fixed dimension.
        self.ctx_builders    = nn.ModuleDict()
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in connect_list:
            in_ch = self.channels[f_size]
            self.ctx_builders[f_size] = CompositeContextBuilder(
                out_dim     = in_ch,
                arcface_dim = arcface_dim,
                farl_dim    = farl_dim,
                farl_tokens = farl_tokens,
                use_farl    = use_farl,
            )
            self.fuse_convs_dict[f_size] = IDCrossAttentionSFT(
                in_ch  = in_ch,
                out_ch = in_ch,
                nhead  = id_nhead,
            )
 
        print(f"[CodeRefFormerV5] use_farl={use_farl}, arcface_dim={arcface_dim}, "
              f"ctx_len={'198' if use_farl else '1'}, "
              f"id_nhead={id_nhead}, connect_list={connect_list}")
 
    def forward(
        self,
        x,
        w:          float = 0,
        detach_16:  bool  = True,
        code_only:  bool  = False,
        adain:      bool  = False,
        x_ref:      Optional[dict] = None,
        id_embs:    Optional[Tensor] = None,   # (B, N, arcface_dim)
        farl_embs:  Optional[Tensor] = None,   # (B, N, 197, farl_dim) or None
        **kwargs,
    ):
        # ---- Encoder ----
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f] for f in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        lq_feat = x
 
        # ---- Pool multi-reference embeddings → single vector per sample ----
        arcface_emb = None
        farl_seq    = None
 
        if id_embs is not None:
            # Attention pool: pick the most identity-relevant reference
            first_res   = self.connect_list[0]
            arcface_emb = self.id_pool(id_embs, enc_feat_dict[first_res])  # (B, 512)
 
        if self.use_farl and farl_embs is not None:
            # Mean pool across N references: (B, N, 197, D) → (B, 197, D)
            farl_seq = farl_embs.mean(dim=1)
 
        # ---- Transformer — pure SA, no reference (identical to V1) ----
        pos_emb   = self.position_emb.unsqueeze(1).repeat(1, lq_feat.shape[0], 1)
        query_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
 
        logits = self.idx_pred_layer(query_emb)
        logits = logits.permute(1, 0, 2)
 
        if code_only:
            return logits, lq_feat
 
        # ---- Quantisation ----
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx   = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat   = self.quantize.get_codebook_feat(
            top_idx, shape=[lq_feat.shape[0], 16, 16, 256])
 
        if detach_16:
            quant_feat = quant_feat.detach()
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
 
        # ---- Generator + composite ID-SFT ----
        x         = quant_feat
        fuse_list = [self.fuse_generator_block[f] for f in self.connect_list]
 
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in fuse_list:
                f_size = str(x.shape[-1])
                if w > 0:
                    # Build composite context for this resolution level:
                    # projects arcface_emb (and farl_seq) to C_level space
                    ctx = None
                    if arcface_emb is not None:
                        ctx = self.ctx_builders[f_size](arcface_emb, farl_seq)
                        # ctx: (1, B, C_level) or (198, B, C_level)
 
                    enc_ref = (x_ref[f_size].detach()
                               if x_ref is not None and f_size in x_ref
                               else None)
 
                    x = self.fuse_convs_dict[f_size](
                        enc_feat_dict[f_size].detach(),
                        x,
                        w,
                        ctx           = ctx,
                        enc_ref_feats = enc_ref,
                    )
 
        return x, logits, lq_feat
    

class TransformerSALayerKVExtractor(nn.Module):
    """Thin wrapper over TransformerSALayer that also returns the raw K and V
    projections computed inside self-attention, without touching the SA logic.
 
    K and V are extracted via a forward pre-hook on nn.MultiheadAttention so
    that we read the *projected* K/V (after in_proj_weight is applied) before
    the attention scores are computed.  The hook stores them in instance
    attributes; they are valid for the duration of the forward call.
    """
 
    def __init__(self, sa_layer: TransformerSALayer):
        super().__init__()
        self.layer = sa_layer
        self._k: Optional[Tensor] = None
        self._v: Optional[Tensor] = None
 
        def _kv_hook(module, args, kwargs):
            # args = (query, key, value, ...)  before any internal projection
            # We re-apply in_proj_weight ourselves to get projected K and V.
            q_in = args[0]
            k_in = args[1] if len(args) > 1 else kwargs.get('key',   q_in)
            v_in = args[2] if len(args) > 2 else kwargs.get('value', q_in)
            W = module.in_proj_weight          # (3C, C)
            b = module.in_proj_bias            # (3C,) or None
            C = q_in.shape[-1]
 
            self._k = F.linear(
                k_in,
                W[C : 2 * C],
                b[C : 2 * C] if b is not None else None,
            )  # (S, B, C)
            self._v = F.linear(
                v_in,
                W[2 * C :],
                b[2 * C :] if b is not None else None,
            )  # (S, B, C)
 
        self.layer.self_attn.register_forward_pre_hook(
            _kv_hook, with_kwargs=True
        )
 
    def forward(
        self,
        tgt: Tensor,
        query_pos: Optional[Tensor] = None,
    ):
        out = self.layer(tgt, query_pos=query_pos)
        return out, self._k, self._v
    

class TransformerSACALayerV3(nn.Module):
    """Self-attention + cross-attention layer for the LQ transformer.
 
    Cross-attention attends to K/V coming from the frozen HQ transformer.
    The contribution is gated by a learnable scalar initialised at 0 so the
    model starts as a plain SA transformer and gradually opens up to the
    reference signal.
 
    AdaIN on K/V (adain_ref=True) aligns the statistical distribution of the
    HQ K/V with the LQ K/V computed at the same layer before the CA step,
    which helps when lighting/colour differ between LQ and reference.
    AdaIN can be applied globally (per-channel) or per attention head.
    """
 
    def __init__(
        self,
        embed_dim: int,
        nhead: int = 8,
        dim_mlp: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        adain_ref: bool = False,
        adain_per_head: bool = False,
    ):
        super().__init__()
        self.nhead          = nhead
        self.head_dim       = embed_dim // nhead
        self.adain_ref      = adain_ref
        self.adain_per_head = adain_per_head
 
        # Self-attention (identical to TransformerSALayer)
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
 
        # Cross-attention
        self.cross_attn  = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.norm_ca     = nn.LayerNorm(embed_dim)
        self.dropout_ca  = nn.Dropout(dropout)
 
        # Learnable gate — starts at 0 (tanh(0) = 0) so CA is initially silent
        self.ca_gate = nn.Parameter(torch.zeros(1))
 
        # Optional LQ K/V projections used as AdaIN style source
        if adain_ref:
            self.lq_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.lq_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
 
        # FFN
        self.linear1  = nn.Linear(embed_dim, dim_mlp)
        self.linear2  = nn.Linear(dim_mlp, embed_dim)
        self.dropout  = nn.Dropout(dropout)
 
        # Layer norms
        self.norm1    = nn.LayerNorm(embed_dim)
        self.norm2    = nn.LayerNorm(embed_dim)
 
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
 
        self.activation = _get_activation_fn(activation)
 
    # ------------------------------------------------------------------
    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos
 
    # ------------------------------------------------------------------
    def _adain_kv(
        self,
        ref_k: Tensor,
        ref_v: Tensor,
        lq_k:  Tensor,
        lq_v:  Tensor,
    ):
        """Align ref K/V distribution to match LQ K/V distribution.
 
        All tensors are (S, B, C).
        """
        if self.adain_per_head:
            return self._adain_kv_per_head(ref_k, ref_v, lq_k, lq_v)
 
        def adain_1d(content: Tensor, style: Tensor) -> Tensor:
            # stats over sequence dimension, keeping (B, C) shape
            s_mean = style.mean(dim=0, keepdim=True)
            s_std  = style.std( dim=0, keepdim=True).clamp(min=1e-5)
            c_mean = content.mean(dim=0, keepdim=True)
            c_std  = content.std( dim=0, keepdim=True).clamp(min=1e-5)
            return (content - c_mean) / c_std * s_std + s_mean
 
        return adain_1d(ref_k, lq_k), adain_1d(ref_v, lq_v)
 
    def _adain_kv_per_head(
        self,
        ref_k: Tensor,
        ref_v: Tensor,
        lq_k:  Tensor,
        lq_v:  Tensor,
    ):
        """Per-head AdaIN — respects the multi-head structure."""
 
        def adain_ph(content: Tensor, style: Tensor) -> Tensor:
            S_c, B, C = content.shape
            S_s       = style.shape[0]
            c = content.view(S_c, B, self.nhead, self.head_dim)
            s = style.view(  S_s, B, self.nhead, self.head_dim)
            s_mean = s.mean(0, keepdim=True)
            s_std  = s.std( 0, keepdim=True).clamp(min=1e-5)
            c_mean = c.mean(0, keepdim=True)
            c_std  = c.std( 0, keepdim=True).clamp(min=1e-5)
            return ((c - c_mean) / c_std * s_std + s_mean).view(S_c, B, C)
 
        return adain_ph(ref_k, lq_k), adain_ph(ref_v, lq_v)
 
    # ------------------------------------------------------------------
    def forward(
        self,
        tgt:       Tensor,
        ref_kv:    None,
        tgt_mask:               Optional[Tensor]  = None,
        tgt_key_padding_mask:   Optional[Tensor]  = None,
        query_pos:              Optional[Tensor]  = None,
    ) -> Tensor:
 
        # ---- Self-attention ----
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
 
        # Capture LQ K/V before SA (used as AdaIN style source)
        if self.adain_ref and ref_kv is not None:
            lq_k = self.lq_k_proj(tgt2)   # (S, B, C)
            lq_v = self.lq_v_proj(tgt2)
 
        tgt2 = self.self_attn(
            q, k, value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
 
        # ---- Cross-attention (if reference K/V available) ----
        if ref_kv is not None:
            ref_k, ref_v = ref_kv
 
            if self.adain_ref:
                ref_k, ref_v = self._adain_kv(ref_k, ref_v, lq_k, lq_v)
 
            tgt2 = self.norm_ca(tgt)
            q    = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.cross_attn(q, ref_k, value=ref_v)[0]
            tgt  = tgt + self.ca_gate.tanh() * self.dropout_ca(tgt2)
 
        # ---- FFN ----
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt  = tgt + self.dropout2(tgt2)
        return tgt
