import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from einops import rearrange

from basicsr.archs.arch_util import flow_warp, resize_flow
from basicsr.losses.gan_loss import r1_penalty
from basicsr.utils import get_root_logger, tensor2img

from .base_model import BaseModel
from .networks import define_generator, define_discriminator, define_id_net
from .networks.vqgan_arch import VQAutoEncoder
from .networks.face_descriptor_nets.farl_wrapper import (
    farl_preprocess_neg1to1, farl_visual_tokens)
from .optim import define_criterion, define_lr_schedule
from utils.net_utils import WarmupTrainer


# =============================================================================
# Module-level helpers (occlusion mask + temporal / identity losses)
# =============================================================================

def _compute_occlusion_mask(flow_fwd, flow_bwd, threshold=1.0):
    """Forward-backward flow consistency mask.

    Args:
        flow_fwd: (B, 2, H, W) — flow from frame t to t+1
        flow_bwd: (B, 2, H, W) — flow from frame t+1 to t
        threshold: relative error threshold
    Returns:
        mask: (B, 1, H, W) float — 1 = visible, 0 = occluded
    """
    flow_bwd_warped = flow_warp(flow_bwd, flow_fwd.permute(0, 2, 3, 1))
    consistency     = flow_fwd + flow_bwd_warped
    error           = (consistency ** 2).sum(dim=1, keepdim=True)
    flow_mag        = (flow_fwd ** 2).sum(dim=1, keepdim=True) + \
                      (flow_bwd_warped ** 2).sum(dim=1, keepdim=True)
    return (error < threshold * flow_mag + 0.5).float()


def _temporal_consistency_loss(curr_feat, prev_feat, flow_fwd, flow_bwd, criterion):
    """Flow-warped temporal consistency loss with occlusion masking.

    Args:
        curr_feat: (B, C, H, W) — decoder features at frame t
        prev_feat: (B, C, H, W) — decoder features at frame t-1
        flow_fwd:  (B, 2, H, W) — flow t-1 → t
        flow_bwd:  (B, 2, H, W) — flow t → t-1
        criterion: loss function (e.g. Charbonnier or L1)
    Returns:
        Scalar loss value.
    """
    H, W = curr_feat.shape[-2:]
    if flow_fwd.shape[-2:] != (H, W):
        flow_fwd = resize_flow(flow_fwd, 'shape', [H, W])
        flow_bwd = resize_flow(flow_bwd, 'shape', [H, W])

    warp_feat = flow_warp(prev_feat, flow_fwd.permute(0, 2, 3, 1))
    mask      = _compute_occlusion_mask(flow_fwd, flow_bwd)
    diff      = (curr_feat - warp_feat) * mask
    n_valid   = mask.sum().clamp(min=1.0)
    return criterion(diff, torch.zeros_like(diff)) * diff.numel() / n_valid

def _token_kl_consistency_loss(logits, B, T, tau=2.0):
    """Token distribution consistency loss across consecutive frames.

    Computes Jensen-Shannon divergence between soft token distributions of
    consecutive frames. Reshape ensures no cross-batch pairs are compared.

    Args:
        logits: (B*T, HW, codebook_size) — raw transformer logits
        B:      batch size
        T:      number of frames per clip
        tau:    softmax temperature (higher = softer = denser gradients)
    Returns:
        Scalar loss.
    """
    probs = F.softmax(logits / tau, dim=-1)              # (B*T, HW, codebook_size)
    probs = rearrange(probs, '(b t) hw c -> b t hw c', b=B, t=T)

    probs_curr = probs[:, 1:]   # (B, T-1, HW, C)
    probs_prev = probs[:, :-1]  # (B, T-1, HW, C)

    m      = 0.5 * (probs_curr + probs_prev)
    js_div = 0.5 * (
        F.kl_div(m.log().clamp(min=-100), probs_curr, reduction='none').sum(-1) +
        F.kl_div(m.log().clamp(min=-100), probs_prev, reduction='none').sum(-1)
    )  # (B, T-1, HW)

    return js_div.mean()

def _sigma_ids_loss(pred_frames, id_net, T):
    """Temporal identity variance loss (sigma_IDS).

    Penalises variance of cosine similarity to the first frame across the clip,
    encouraging stable identity throughout the sequence.

    Args:
        pred_frames: (B*T, C, H, W) float [-1, 1]
        id_net:      frozen ArcFace model
        T:           temporal window length
    Returns:
        Scalar loss.
    """
    B = pred_frames.shape[0] // T
    emb      = id_net(pred_frames)                                         # (B*T, 512)
    emb      = rearrange(emb, '(b t) d -> b t d', b=B, t=T)
    anchor   = emb[:, 0:1].expand_as(emb)                                 # (B, T, 512)
    cos_sim  = F.cosine_similarity(emb, anchor, dim=2)                    # (B, T)
    return cos_sim.std(dim=1).mean()


# =============================================================================
# Model
# =============================================================================

class CodeFormerVSRModel(BaseModel):
    """Video face super-resolution model built on a frozen CodeFormer backbone.

    Only the temporal modules (hq_encoder, align_net, blend_net, CFA layers)
    are trained.  The single-image backbone (lq_encoder, transformer, quantize,
    generator / CFT blocks) remains frozen throughout.

    The class structure mirrors CodeFormerJointRefModel:
        __init__ → set_network → config_training (→ set_criterion)
        feed_data / train / infer / save
    """

    def __init__(self, opt):
        super().__init__(opt)

        if self.verbose:
            self.logger.info('{} CodeFormerVSR Model Info {}'.format('=' * 20, '=' * 20))
            self.logger.info('Model: {}'.format(opt['model']['name']))

        self.set_network()

        if self.is_train:
            self.config_training()

        self.w_scale = self.opt['model']['generator']['backbone_cfg'].get('w_scale', 1)
        self.detach_16 = self.opt['model']['generator']['backbone_cfg'].get('detach_16', True)
        self.early_feat = self.opt['model']['generator']['backbone_cfg'].get('early_feat', False)

        # Reference conditioning flags — mirrors CodeFormerJointRefModel
        # conditional=False for blind model, True for reference-based
        self.conditional   = self.opt['model']['generator']['backbone_cfg'].get('conditional', False)
        self.return_ref_kv = self.opt['model']['generator']['backbone_cfg'].get('return_ref_kv', False)
        self.ref_augment   = self.opt['dataset']['train'].get('ref_augment', False)

        # Sequential running
        self.sequential_refs = self.opt['model'].get('sequential_refs', False)

        # Cache for reference features — computed once per clip, reused for all frames
        # None when blind (no reference), populated in feed_data when ref is present
        self._ref_feat_cache    = None  # x_ref: dict of HQ encoder features
        self._id_embs_cache     = None  # id_embs: (B, N, 512)
        self._farl_embs_cache   = None  # farl_embs: (B, N, 197, 768)

        # Set up embedding networks (ArcFace + FaRL) — runs at both train and infer time
        # so that feed_data() can always reference self.net_ID and self.net_FaRL safely.
        self._setup_embedding_nets()# Set up embedding networks (ArcFace + FaRL) — runs at both train and infer time

    # ------------------------------------------------------------------
    # Network setup
    # ------------------------------------------------------------------
    def _setup_embedding_nets(self):
        """Initialise ArcFace (net_ID) and FaRL (net_FaRL) encoder networks.
 
        Called unconditionally from __init__ so that both attributes exist at
        both train and inference time.  feed_data() checks them with simple
        ``is not None`` guards rather than hasattr(), avoiding AttributeErrors.
 
        net_ID path:   opt['model']['id_network_path']  (optional)
        net_FaRL path: opt['model']['generator']['farl_path']  (optional)
 
        Either network stays None when its path is not configured.
        """
        # ── ArcFace identity encoder ─────────────────────────────────────────
        # At train time set_criterion() also reads the path from opt['train']['id_crit'],
        # but we load from opt['model'] here so the same network is available at
        # inference without a training config section.
        self.net_ID = None
        id_net_path = self.opt['model']['generator'].get('id_network_path', None)
        if id_net_path:
            self.net_ID = define_id_net(id_net_path).to(self.device)
            self.net_ID.eval()
            for p in self.net_ID.parameters():
                p.requires_grad = False
            if self.verbose:
                self.logger.info(f'net_ID (ArcFace) loaded from: {id_net_path}')
 
        # ── FaRL CLIP encoder ─────────────────────────────────────────────────
        # Loaded when farl_path is set, regardless of ref_augment / is_train.
        self.net_FaRL = None
        farl_path = self.opt['model']['generator'].get('farl_path', None)
        if farl_path:
            self._load_farl_encoder()

    def set_network(self):
        # Load HQ VQGAN encoder
        if self.opt['model'].get('network_vqgan', None) is not None:
            vqgan_path   = self.opt['model']['network_vqgan']['load_path']
            hq_vqgan_fix = VQAutoEncoder(**self.opt['model']['network_vqgan'])
            self.load_network(hq_vqgan_fix, vqgan_path, strict_load=True)

            # Keep encoder + quantizer at model level (GT token indices for CE loss)
            self.hq_encoder  = hq_vqgan_fix.encoder.to(self.device)
            self.hq_quantize = hq_vqgan_fix.quantize.to(self.device)

            hq_encoder_state_dict = {
                k: v.cpu() for k, v in hq_vqgan_fix.encoder.state_dict().items()
            }

            del hq_vqgan_fix

            self.hq_encoder.eval()
            self.hq_quantize.eval()
            for p in self.hq_encoder.parameters():
                p.requires_grad = False
            for p in self.hq_quantize.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError('network_vqgan config is required.')
        # Inject HQ encoder's state dict into the configuration
        self.opt['model']['generator']['hq_encoder_state_dict'] = hq_encoder_state_dict

        # Generator (CodeFormer VSR variant with temporal modules)
        self.net_G = define_generator(self.opt).to(self.device)

        # Remove the state_dict from opt after construction — it is a large
        # tensor dict and should not persist in the config object.
        self.opt['model']['generator'].pop('hq_encoder_state_dict', None)

        if self.verbose:
            self.logger.info('Generator: {}\n'.format(
                self.opt['model']['name']) + str(self.net_G))

        # NEW:
        load_path_G = self.opt['model']['g_load_path']
        if load_path_G:
            ckpt = torch.load(load_path_G, map_location='cpu')
            state_dict = (
                ckpt.get('params_ema') or
                ckpt.get('params')     or
                ckpt.get('state_dict') or
                ckpt
            )

            # Auto-detect whether this is a full video checkpoint (keys start
            # with 'backbone.') or a single-image backbone checkpoint (keys
            # do not have that prefix).  Remap if needed so the keys match
            # net_G's layout in both cases.
            first_key = next(iter(state_dict))
            if not first_key.startswith('backbone.'):
                # Single-image backbone checkpoint — add prefix before loading
                state_dict = {f'backbone.{k}': v for k, v in state_dict.items()}
                if self.verbose:
                    self.logger.info(
                        f'Detected single-image backbone checkpoint — '
                        f'remapping keys with "backbone." prefix.'
                    )

            missing, unexpected = self.net_G.load_state_dict(
                state_dict, strict=False)
            if self.verbose:
                self.logger.info(
                    f'Loaded generator from: {load_path_G} '
                    f'(missing={len(missing)}, unexpected={len(unexpected)})'
                )

                if len(missing) > 0:
                        print(f"Missing keys: {missing}.")
                    
                if len(unexpected) > 0:
                    print(f"Unexpected: {unexpected}.")

        # WarmupTrainer (freeze / unfreeze per-module schedule)
        self.warmup_trainer = None
        if self.is_train:
            warmup_cfg = self.opt['train'].get('warmup', None)
            if warmup_cfg and warmup_cfg.get('modules'):
                self.warmup_trainer = WarmupTrainer(self.net_G, warmup_cfg)
                if self.verbose:
                    self.logger.info('WarmupTrainer initialised.')

        self.init_ema_network()

        # Discriminator
        if self.is_train and 'gan_crit' in self.opt['train']:
            net_D_opt  = self.opt['model']['discriminator']
            net_D_name = self.opt['model']['d_name'].lower()
            self.net_D = define_discriminator(net_D_opt, net_D_name=net_D_name).to(self.device)

            load_path_D = self.opt['model'].get('d_load_path', None)
            if load_path_D:
                self.load_network(self.net_D, load_path_D, strict_load=False)
                if self.verbose:
                    self.logger.info('Loaded discriminator from: {}'.format(load_path_D))
        else:
            self.net_D = None

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------
    def config_training(self):
        self.set_criterion()

        # Mixed precision (AMP) — see BaseModel._init_amp
        self._init_amp()

        # Generator optimiser — all parameters registered; WarmupTrainer
        # controls requires_grad, so frozen params are skipped automatically.
        self.optim_G = optim.AdamW(
            self.net_G.parameters(),
            lr=self.opt['train']['generator']['lr'],
            weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
            betas=(
                self.opt['train']['generator'].get('beta1', 0.9),
                self.opt['train']['generator'].get('beta2', 0.999),
            ),
        )
        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G,
        )

        if self.gan_crit is not None:
            discriminator_opt = self.opt['train']['discriminator']
            self.optim_D = optim.AdamW(
                self.net_D.parameters(),
                lr=discriminator_opt['lr'],
                weight_decay=discriminator_opt.get('weight_decay', 0),
                betas=(
                    discriminator_opt.get('beta1', 0.9),
                    discriminator_opt.get('beta2', 0.999),
                ),
            )
            self.sched_D = define_lr_schedule(
                discriminator_opt.get('lr_schedule'), self.optim_D,
            )

            gan_crit_opt          = self.opt['train']['gan_crit']
            self.r1_reg_weight    = gan_crit_opt['r1_reg_weight']
            self.net_d_iters      = gan_crit_opt.get('net_d_iters', 1)
            self.net_d_init_iters = gan_crit_opt.get('net_d_init_iters', 0)
            self.net_d_reg_every  = gan_crit_opt['net_d_reg_every']
            self.net_d_start_iters = gan_crit_opt['net_d_start_iters']
        else:
            self.net_d_iters       = 1
            self.net_d_init_iters  = 0
            self.net_d_start_iters = 0

    def set_criterion(self):
        train_opt = self.opt['train']

        # Pixel reconstruction loss
        self.pix_crit = define_criterion(train_opt.get('pixel_crit'))

        # Perceptual / feature loss
        self.feat_crit = define_criterion(train_opt.get('feature_crit'))
        if self.feat_crit is not None:
            self.feat_crit = self.feat_crit.to(self.device)

        # Identity loss — IDS (per frame, vs GT)
        self.id_crit = define_criterion(train_opt.get('id_crit'))
        if self.id_crit is not None:
            id_net_path  = train_opt['id_crit']['id_network_path']
            self.id_crit_hard = train_opt['id_crit'].get('hard_loss', False)
            
            # Define and load ID net
            if hasattr(self, 'net_ID') and self.net_ID is None:
                self.net_ID = define_id_net(id_net_path).to(self.device)
            else:
                print("Skipping net_ID initialization: already initialized.")

        # Temporal consistency loss (flow-warped feature matching)
        self.temporal_crit = define_criterion(train_opt.get('temporal_crit'))

        # Temporal KL on tokens
        self.token_kl_weight  = train_opt.get('token_kl_crit', {}).get('weight', 0.0)
        self.token_kl_tau    = train_opt.get('token_kl_crit', {}).get('tau', 2.0)

        # sigma_IDS — temporal identity variance loss
        self.sigma_id_weight = train_opt.get('sigma_id_crit', {}).get('weight', 0.0)

        # GAN loss
        self.gan_crit = define_criterion(train_opt.get('gan_crit'))

        # Adaptive GAN update policy
        self.use_adaptive_weight       = train_opt.get('use_adaptive_weight', False)
        self.scale_adaptive_gan_weight = train_opt.get('scale_adaptive_gan_weight', 0.8) 

        # CodeFormer-specific losses
        self.hq_feat_loss      = train_opt.get('use_hq_feat_loss', False)
        self.feat_loss_weight  = train_opt.get('feat_loss_weight', 1.0)
        self.cross_entropy_loss = train_opt.get('cross_entropy_loss', False)
        self.entropy_loss_weight = train_opt.get('entropy_loss_weight', 0.5)

        self.fix_generator   = self.warmup_trainer.is_module_frozen('backbone.generator') \
                               if self.warmup_trainer else False
        self.fix_transformer = self.warmup_trainer.is_module_frozen('backbone.ft_layers') \
                               if self.warmup_trainer else False
        self.logger.info(
            f'fix_generator: {self.fix_generator}. fix_transformer: {self.fix_transformer}'
        )

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def feed_data(self, data):
        self.lr = data['lr'].to(self.device)   # (B, T, C, H, W)
        self.gt = data.get('gt', None)
        if self.gt is not None:
            self.gt = self.gt.to(self.device)

        # Pre-computed GT ArcFace embeddings (optional — from LMDB metadata)
        self.id_embs_gt = data.get('id_embs', None)
        if self.id_embs_gt is not None:
            self.id_embs_gt = self.id_embs_gt.to(self.device)

        # GT codebook token indices (optional — pre-computed offline)
        self.idx_gt = data.get('latent_gt', None)
        if self.idx_gt is not None:
            self.idx_gt = self.idx_gt.to(self.device)

        # --- Reference images (optional — reference-based mode only) ---
        self.ref = data.get('ref', None)
        if isinstance(self.ref, torch.Tensor) and self.ref.numel() > 0:
            self.ref = self.ref.to(self.device)
        else:
            self.ref = None

        # --- Reference embeddings from cache (pre-loaded from LMDB) ---
        self.id_embeddings   = data.get('id_embeddings', None)
        if isinstance(self.id_embeddings, dict):
            for k in self.id_embeddings:
                self.id_embeddings[k] = self.id_embeddings[k].to(self.device)

        self.farl_embeddings = data.get('farl_embeddings', None)
        if isinstance(self.farl_embeddings, dict):
            for k in self.farl_embeddings:
                self.farl_embeddings[k] = self.farl_embeddings[k].to(self.device)

        # --- Online ref embedding computation when ref_augment is active ---
        # Identical to CodeFormerJointRefModel.feed_data — but we CACHE the result
        # so it is not recomputed for every temporal frame in this clip.
        if ( self.ref_augment and self.is_train and self.ref is not None ) or \
            ( not self.is_train and self.farl_embeddings is None ) :
            B, N, C, H, W = self.ref.shape
            ref_flat = self.ref.view(B * N, C, H, W)

            if self.net_ID is not None and self.id_embeddings is None:
                with torch.no_grad():
                    arc_embs = self.net_ID(ref_flat)
                arc_embs = arc_embs.view(B, N, 512).unsqueeze(2)
                self.id_embeddings = {'refs': arc_embs}

            # FaRL — CLIP ViT-B/16 preprocessing (224x224) then full transformer
            # forward keeping all 197 tokens (see farl_wrapper for the shared core).
            if self.net_FaRL is not None and self.farl_embeddings is None:
                with torch.no_grad():
                    clip_imgs = farl_preprocess_neg1to1(ref_flat)                 # (B*N, 3, 224, 224)
                    x = farl_visual_tokens(self.net_FaRL.visual, clip_imgs)       # (B*N, 197, 768)
                farl_embs = x.view(B, N, 197, 768)                               # (B, N, 197, 768)
                self.farl_embeddings = {'refs': farl_embs}

        # --- Compute and cache reference features (HQ encoder + id/farl) ---
        # References are per-clip, not per-frame — compute once and reuse.
        # Cache is invalidated when a new batch arrives (ref tensor changes).
        if self.conditional and self.ref is not None and self.w_scale > 0:
            self._ref_feat_cache  = self._extract_reference_features(
                return_ref_kv=self.return_ref_kv
            )
        else:
            self._ref_feat_cache  = None

        if self.id_embeddings is not None and 'refs' in self.id_embeddings:
            self._id_embs_cache   = self.id_embeddings['refs'].squeeze(2)  # (B, N, 512)
        else:
            self._id_embs_cache   = None

        if self.farl_embeddings is not None and 'refs' in self.farl_embeddings:
            self._farl_embs_cache = self.farl_embeddings['refs']            # (B, N, 197, 768)
        else:
            self._farl_embs_cache = None


    # ------------------------------------------------------------------
    # Optical flow (frozen GMFlow, bidirectional)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_flows(self, frames):
        """Compute bidirectional optical flows for all consecutive pairs.

        Args:
            frames: (B, T, C, H, W) in [-1, 1]
        Returns:
            flows_fwd: (B*(T-1), 2, H, W) — flow from t-1 to t
            flows_bwd: (B*(T-1), 2, H, W) — flow from t to t-1
        """
        f_curr = rearrange(frames[:, :-1], 'b t c h w -> (b t) c h w')
        f_next = rearrange(frames[:, 1:],  'b t c h w -> (b t) c h w')
        flows_fwd = self.net_G.flownet(f_next, f_curr)
        flows_bwd = self.net_G.flownet(f_curr, f_next)
        return flows_fwd, flows_bwd

    # ------------------------------------------------------------------
    # GT token indices (for cross-entropy loss)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_gt_token_indices(self, gt_flat):
        """Encode GT frames to codebook token indices via frozen HQ encoder.

        Args:
            gt_flat: (B*T, C, H, W) in [-1, 1]
        Returns:
            idx_gt: (B*T, H*W) long tensor
        """
        z_gt               = self.hq_encoder(gt_flat)
        _, _, quant_stats  = self.hq_quantize(z_gt)
        idx_gt             = quant_stats['min_encoding_indices']
        return idx_gt.view(gt_flat.shape[0], -1)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train(self, data, iter, accum_steps=1, is_last_accum=True):
        """Single mini-batch training step for video face SR."""
        self.feed_data(data)
        B, T, C, H, W = self.lr.shape

        # Track whether any optimizer stepped this iter (for AMP scaler.update)
        stepped = False

        # Build forward dict — reference features already cached in feed_data,
        # so they are not recomputed for each frame here.
        self.net_G.train()
        if self.net_D is not None:
            self.set_requires_grad(self.net_D, False)

        # Build forward dict
        train_dict = {
            'w':          self.w_scale,
            'detach_16':  self.detach_16,
            'early_feat': self.early_feat
        }
        if self._ref_feat_cache is not None:
            train_dict['x_ref']      = self._ref_feat_cache
        if self._id_embs_cache is not None:
            train_dict['id_embs']    = self._id_embs_cache
        if self._farl_embs_cache is not None:
            train_dict['farl_embs']  = self._farl_embs_cache
 
        # Forward pass — net_G returns (output, logits, lq_feat, gen_feat_dict)
        # when detach_16=True and early_feat=True (same flags as CodeFormer image).
        # Frozen backbone (encoder/generator) runs in reduced precision under
        # autocast; outputs are cast back to fp32 for stable loss computation.
        with self._amp_autocast():
            self.output, self.logits, self.lq_feat, self.gen_feat_dict = self.net_G(
                self.lr, **train_dict)
        if self.use_amp:
            if self.output is not None:  self.output  = self.output.float()
            if self.logits is not None:  self.logits  = self.logits.float()
            if self.lq_feat is not None: self.lq_feat = self.lq_feat.float()
            if isinstance(self.gen_feat_dict, dict):
                self.gen_feat_dict = {k: (v.float() if torch.is_tensor(v) else v)
                                      for k, v in self.gen_feat_dict.items()}
        # output:        (B, T, C, H, W)
        # logits:        (B*T, H*W, codebook_size)
        # lq_feat:       (B*T, C, h, w)  — LQ encoder features for hq_feat_loss
        # gen_feat_dict: dict f_size -> (B, T, C, h, w)  — decoder features
 
        # Flatten time dimension for per-frame losses
        if self.output is not None:
            out_flat = rearrange(self.output, 'b t c h w -> (b t) c h w')
        else:
            out_flat = None
        gt_flat  = rearrange(self.gt,     'b t c h w -> (b t) c h w').detach()

        # HQ quantized features (same as CodeFormerJointRefModel)
        if self.hq_feat_loss:
            if self.idx_gt is None:
                self.idx_gt = self._get_gt_token_indices(gt_flat)
            quant_feat_gt = self.net_G.backbone.quantize.get_codebook_feat(
                self.idx_gt, shape=[B * T, 16, 16, 256],
            )
 
        # Debug snapshots
        if iter % 5 == 0 and is_last_accum:
            T = self.lr.shape[1]
            n_frames = min(5, T)
            
            # Concatenate frames horizontally into a single image
            lr_frames  = [tensor2img(self.lr[0, t:t+1],     rgb2bgr=True, min_max=(-1, 1)) for t in range(n_frames)]
            sr_frames  = [tensor2img(self.output[0, t:t+1],  rgb2bgr=True, min_max=(-1, 1)) for t in range(n_frames)]
            gt_frames  = [tensor2img(self.gt[0, t:t+1],     rgb2bgr=True, min_max=(-1, 1)) for t in range(n_frames)]
            
            cv2.imwrite('debug/lr_strip.png',  np.concatenate(lr_frames,  axis=1))
            cv2.imwrite('debug/sr_strip.png',  np.concatenate(sr_frames,  axis=1))
            cv2.imwrite('debug/gt_strip.png',  np.concatenate(gt_frames,  axis=1))
            
            if self.ref is not None and self.ref.numel() != 0:
                temp = tensor2img(self.ref[0:1], rgb2bgr=True, min_max=(-1, 1))
                cv2.imwrite('debug/ref.png', temp)
 
        loss_G    = 0
        self.log_dict = OrderedDict()
 
        if iter % self.net_d_iters == 0 and iter >= self.net_d_init_iters:
 
            # ---- HQ feature loss (codebook) ----
            if not self.fix_transformer and self.hq_feat_loss:
                l_feat_encoder = (
                    torch.mean((quant_feat_gt.detach() - self.lq_feat) ** 2)
                    * self.feat_loss_weight
                )
                loss_G += l_feat_encoder
                self.log_dict['l_feat_encoder'] = l_feat_encoder.item()
 
            # ---- Cross-entropy token prediction ----
            if self.cross_entropy_loss:
                l_ce = F.cross_entropy(
                    self.logits.permute(0, 2, 1),   # (B*T, codebook_size, H*W)
                    self.idx_gt,
                ) * self.entropy_loss_weight
                loss_G += l_ce
                self.log_dict['l_ce'] = l_ce.item()

            # ---- Token KL consistency loss ----
            if self.token_kl_weight > 0 and T > 1:
                l_token_kl = self.token_kl_weight * _token_kl_consistency_loss(
                    self.logits, B=B, T=T, tau=self.token_kl_tau,
                )
                loss_G += l_token_kl
                self.log_dict['l_token_kl'] = l_token_kl.item()
                        
            # ---- Pixel loss ----
            if self.pix_crit is not None:
                pix_w   = self.opt['train']['pixel_crit'].get('weight', 1)
                l_pix_G = pix_w * self.pix_crit(out_flat, gt_flat)
                loss_G += l_pix_G
                self.log_dict['l_pix_G'] = l_pix_G.item()
 
            # ---- Perceptual / feature loss ----
            if self.feat_crit is not None:
                feat_w   = self.opt['train']['feature_crit'].get('weight', 1)
                l_feat_G = feat_w * self.feat_crit(out_flat, gt_flat)
                loss_G += l_feat_G
                self.log_dict['l_feat_G'] = l_feat_G.item()
 
            # ---- Identity loss (IDS, per frame vs GT) ----
            if self.id_crit is not None:
                # GT embeddings — use pre-computed LMDB cache when available,
                # otherwise run ArcFace on GT frames on-the-fly.
                if self.id_embs_gt is not None:
                    gt_id_emb = rearrange(
                        self.id_embs_gt, 'b t d -> (b t) d',
                    )                                           # (B*T, 512)
                else:
                    with torch.no_grad():
                        gt_id_emb = self.net_ID(gt_flat)       # (B*T, 512)

                hr_id_emb = self.net_ID(out_flat)               # (B*T, 512)
                l_id_G    = self.id_crit(hr_id_emb, gt_id_emb.detach())

                # Hard loss: also penalize distance to reference identity embeddings.
                # Mirrors CodeFormerJointRefModel — averages N refs → single embedding,
                # then computes one additional id_crit term per clip.
                if self.id_crit_hard and self._id_embs_cache is not None:
                    # _id_embs_cache: (B, N, 512) — average over N references
                    ref_id_emb = self._id_embs_cache.mean(dim=1)   # (B, 512)
                    # Expand to match (B*T, 512) so id_crit sees paired rows
                    ref_id_emb = ref_id_emb.unsqueeze(1).expand(
                        -1, T, -1).reshape(B * T, -1)              # (B*T, 512)
                    l_id_G    += self.id_crit(hr_id_emb, ref_id_emb.detach())
                elif self.id_crit_hard and self.ref_augment and self.ref is not None:
                    # ref_augment mode — embeddings not pre-cached, compute on-the-fly
                    ref_flat_id = self.ref.view(B * self.ref.shape[1], C, H, W)
                    with torch.no_grad():
                        ref_id_emb = self.net_ID(ref_flat_id)      # (B*N, 512)
                    ref_id_emb = ref_id_emb.view(B, -1, 512).mean(dim=1)  # (B, 512)
                    ref_id_emb = ref_id_emb.unsqueeze(1).expand(
                        -1, T, -1).reshape(B * T, -1)              # (B*T, 512)
                    l_id_G    += self.id_crit(hr_id_emb, ref_id_emb.detach())
                id_w   = self.opt['train']['id_crit'].get('weight', 1)
                l_id_G *= id_w
                loss_G += l_id_G
                self.log_dict['l_id_G'] = l_id_G.item()
 
            # ---- sigma_IDS — temporal identity variance ----
            if self.sigma_id_weight > 0 and self.net_ID is not None:
                l_sigma_id = self.sigma_id_weight * _sigma_ids_loss(
                    out_flat, self.net_ID, T,
                )
                loss_G += l_sigma_id
                self.log_dict['l_sigma_id_G'] = l_sigma_id.item()

            # ---- Temporal consistency loss ----
            if self.temporal_crit is not None and T > 1:
                with torch.no_grad():
                    # Compute flows from GT frames — (B*(T-1), 2, H, W) each
                    flows_fwd, flows_bwd = self._compute_flows(self.gt)

                # output: (B, T, C, H, W) — predicted output frames
                prev_out = rearrange(self.output[:, :-1], 'b t c h w -> (b t) c h w')
                curr_out = rearrange(self.output[:, 1:],  'b t c h w -> (b t) c h w')

                l_temp = _temporal_consistency_loss(
                    curr_out, prev_out, flows_fwd, flows_bwd, self.temporal_crit,
                )
                temporal_w  = self.opt['train']['temporal_crit'].get('weight', 1)
                l_temp     *= temporal_w
                loss_G     += l_temp
                self.log_dict['l_temporal_G'] = l_temp.item()
 
            # ---- GAN loss (w/ optional adaptive weight) ----
            if self.gan_crit is not None:
                fake_g_pred = self.net_D(self.output)
                l_gan_G     = self.gan_crit(fake_g_pred, True, is_critic_update=False)
                
                if self.use_adaptive_weight:
                    # Reconstruct loss for gradient-norm balancing
                    # Use pixel + perceptual contributions if available.
                    recon_loss = (
                        (l_pix_G  if self.pix_crit  is not None else 0) +
                        (l_feat_G if self.feat_crit is not None else 0)
                    )
                    # Last trainable generator layer for gradient-norm balancing.
                    # Only defined for the ConvGRU blend_net; resolve the last
                    # conv weight defensively so other backbones (or a frozen /
                    # absent blend_net) fall back to the fixed GAN weight below.
                    last_layer = None
                    blend_net = getattr(self.net_G, 'blend_net', None)
                    if blend_net is not None:
                        inner = getattr(blend_net, 'delta_net', None) \
                            or getattr(blend_net, 'net', None)
                        if inner is not None:
                            for m in reversed(list(inner)):
                                if isinstance(m, torch.nn.Conv2d):
                                    last_layer = m.weight
                                    break
                    if last_layer is not None and last_layer.requires_grad and recon_loss != 0:
                        d_weight = self._calculate_adaptive_weight(
                            recon_loss, l_gan_G, last_layer, disc_weight_max=1.0,
                        )
                        d_weight *= self.scale_adaptive_gan_weight
                        self.log_dict['d_weight'] = d_weight.item()
                        l_gan_G  *= d_weight
                    else:
                        # Layer frozen during warmup — fall back to fixed weight
                        l_gan_G *= self.opt['train']['gan_crit'].get('weight', 1)
                else:
                    l_gan_G *= self.opt['train']['gan_crit'].get('weight', 1)
 
            # ---- Backprop G ----
            loss_G = loss_G / accum_steps
            self.amp_scaler.scale(loss_G).backward()

            if is_last_accum:
                self.amp_scaler.step(self.optim_G)
                self.optim_G.zero_grad()
                stepped = True
                if self.ema_decay > 0:
                    self.model_ema(decay=self.ema_decay)
 
        # ---- Discriminator update ----
        if self.net_D is not None and iter >= self.net_d_start_iters:
            self.set_requires_grad(self.net_D, True)
            if is_last_accum:
                self.optim_D.zero_grad()
 
            fake_d_pred = self.net_D(self.output.detach())
            real_d_pred = self.net_D(self.gt)
 
            loss_D = (
                self.gan_crit(fake_d_pred, False, is_critic_update=True) +
                self.gan_crit(real_d_pred, True,  is_critic_update=True)
            ) / accum_steps
            self.amp_scaler.scale(loss_D).backward()

            self.log_dict['l_gan_D']     = loss_D.item()
            self.log_dict['real_score']  = real_d_pred.detach().mean().item()
            self.log_dict['fake_score']  = fake_d_pred.detach().mean().item()

            # R1 gradient penalty kept in fp32 (unstable in low precision)
            if self.r1_reg_weight > 0 and is_last_accum \
                    and iter % self.net_d_reg_every == 0:
                self.gt.requires_grad_(True)
                real_pred  = self.net_D(self.gt)
                r1_term    = (
                    r1_penalty(real_pred, self.gt)
                    * self.r1_reg_weight * 0.5 * self.net_d_reg_every
                )
                self.amp_scaler.scale(r1_term).backward()
                self.gt.requires_grad_(False)
                self.log_dict['r1_term_D'] = r1_term.detach().mean().item()

            if is_last_accum:
                self.amp_scaler.step(self.optim_D)
                stepped = True

        # ---- AMP scaler update (once per iter, if an optimizer stepped) ----
        if is_last_accum and stepped:
            self.amp_scaler.update()

        # ---- Warmup step ----
        if is_last_accum and self.warmup_trainer is not None:
            self.warmup_trainer.step(iter)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    # codeformer_video_model.py — în metoda infer()
    def infer(self, data, chunk_size: int = 10, **kwargs):
        """Inference on a video clip with chunked processing to avoid OOM.

        Processes the clip in overlapping chunks to keep VRAM usage bounded.
        Pass chunk_size=None to process the whole clip in a single pass.
        chunk_size=8 is safe for a 3060 at 512x512.
        """
        self.feed_data(data)

        infer_dict = {'w': self.w_scale,
                      'early_feat': False}

        # Reference features are clip-level — safe to keep in VRAM
        if self._ref_feat_cache is not None:
            infer_dict['x_ref']     = self._ref_feat_cache
        if self._id_embs_cache is not None:
            infer_dict['id_embs']   = self._id_embs_cache
        if self._farl_embs_cache is not None:
            infer_dict['farl_embs'] = self._farl_embs_cache

        infer_net = self.net_G_ema if self.ema_decay > 0 else self.net_G

        T = self.lr.shape[1]  # total number of frames
        if chunk_size is None:
            chunk_size = T
        if T <= chunk_size:
            # Short clip — process in one pass (original behaviour)
            with torch.no_grad():
                infer_dict['early_feat'] = False
                sr_frames, *_ = infer_net(self.lr, **infer_dict)
            return {'hr_data': sr_frames.cpu()}
        else:
            # Long clip — process in non-overlapping chunks,
            # carrying GRU hidden state and last decoded frame across boundaries
            all_chunks = []
            h_state = None   # GRU hidden state carried between chunks
            prev_out = None   # last decoded frame carried between chunks
            prev_feats = None

            with torch.no_grad():
                infer_dict['early_feat'] = False
                for start in range(0, T, chunk_size):
                    end      = min(start + chunk_size, T)
                    lr_chunk = self.lr[:, start:end]          # (1, chunk, C, H, W)

                    # Pass carry-in states; receive carry-out states.
                    # Recurrent models (ConvGRUVideoVSR) return a 4-tuple
                    # (outs, h, prev_out, prev_feats); non-recurrent models
                    # (TemporalAttnVideoVSR) return just outs.
                    result = infer_net(
                        lr_chunk,
                        h_init=h_state,
                        prev_out_init=prev_out,
                        prev_feats_init=prev_feats,
                        **infer_dict,
                    )
                    if isinstance(result, tuple):
                        sr_chunk, h_state, prev_out, prev_feats = result
                    else:
                        sr_chunk = result
                    all_chunks.append(sr_chunk.cpu())
                    torch.cuda.empty_cache()

        sr_frames = torch.cat(all_chunks, dim=1)           # reassemble along T dimension
        return {'hr_data': sr_frames}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
        if self.net_D is not None:
            self.save_network(self.net_D, 'D', current_iter)