import cv2
import torch
import torch.optim as optim
from collections import OrderedDict
from os import path as osp

from .base_model import BaseModel

from .networks import *
from .networks.vqgan_arch import VQAutoEncoder
from .networks.face_descriptor_nets.farl_wrapper import (
    farl_preprocess_neg1to1, farl_visual_tokens)
from .optim import define_criterion, define_lr_schedule
from utils.net_utils import WarmupTrainer

from basicsr.losses.gan_loss import r1_penalty 
from torchvision.ops import roi_align

from basicsr.utils import get_root_logger, tensor2img

import torch.nn.functional as F


class CodeFormerJointRefModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        if self.verbose:
            self.logger.info('{} CodeFormerJointRef Model Info {}'.format('=' * 20, '=' * 20))
            self.logger.info('Model: {}'.format(opt['model']['name']))

        self.set_network()

        if self.is_train:
            self.config_training()

        # Ref Conditioning or not
        self.conditional = self.opt['model']['generator']['conditional']
        self.return_ref_kv = self.opt['model']['generator'].get('return_ref_kv', False)
        self.w_scale = self.opt['model']['generator'].get('w_scale', 1)
        self.ref_augment = self.opt['dataset']['train'].get('ref_augment', False)

        # Sequential running
        self.sequential_refs = self.opt['model'].get('sequential_refs', False)

        # Set up embedding networks (ArcFace + FaRL) — runs at both train and infer time
        # so that feed_data() can always reference self.net_ID and self.net_FaRL safely.
        self._setup_embedding_nets()# Set up embedding networks (ArcFace + FaRL) — runs at both train and infer time

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
        id_net_path = self.opt['model']['generator'].get('id_network_path', None)
        if id_net_path and not hasattr(self, 'net_ID'):
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
        # Define and load generator net
        self.net_G = define_generator(self.opt).to(self.device)

        if self.verbose:
            self.logger.info('Generator: {}\n'.format(
                self.opt['model']['g_name']) + str(self.net_G))

        load_path_G = self.opt['model']['g_load_path']
        if load_path_G:
            self.load_network(self.net_G, load_path_G, 
                              self.opt['model']['generator'].get('strict_load', True))
            if self.verbose:
                self.logger.info('Loaded generator from: {}'.format(load_path_G))

        # Load HQ VQ-VAEGAN encoder if needed
        if self.opt['model'].get('network_vqgan', None) is not None:  # Using this network for references
            vqgan_path = self.opt['model']['network_vqgan']['load_path']
            hq_vqgan_fix = VQAutoEncoder(**self.opt['model']['network_vqgan'])
            self.load_network(hq_vqgan_fix, vqgan_path, strict_load=True)

            # Keep only the encoder and quantizer
            self.hq_encoder = hq_vqgan_fix.encoder
            self.hq_quantize = hq_vqgan_fix.quantize

            del hq_vqgan_fix

            # Move to GPU and set to eval mode
            self.hq_encoder = self.hq_encoder.to(self.device)
            self.hq_quantize = self.hq_quantize.to(self.device)
            self.hq_encoder.eval()
            self.hq_quantize.eval()

            # No grad computation
            for param in self.hq_encoder.parameters():
                param.requires_grad = False
            for param in self.hq_quantize.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(f'Shoule have network_vqgan config for references.') 

        # If the generator uses the HQ reference transformer (use_hq_transformer),
        # initialise the frozen HQ transformer with the SA weights from the
        # (just loaded) trainable LQ transformer.
        # This must happen AFTER load_network so any loaded weights are not overwritten.
        if hasattr(self.net_G, 'hq_ft_layers') and hasattr(self.net_G, 'ft_layers'):
            if self.opt['model']['generator'].get('init_hq_ft_from_ft', False):
                if self.verbose:
                    self.logger.info('init_hq_ft_from_ft=True — copying ft_layers SA weights to hq_ft_layers ...')
                for lq_layer, hq_layer in zip(self.net_G.ft_layers, self.net_G.hq_ft_layers):
                    target_sd = hq_layer.layer.state_dict()
                    source_sd = {k: v for k, v in lq_layer.state_dict().items() if k in target_sd}
                    missing, unexpected = hq_layer.layer.load_state_dict(source_sd, strict=False)
                    self.logger.info(f"Loading hq_ft_layers. Missing: {missing}. Unexpected: {unexpected}.")
                if self.verbose:
                    self.logger.info('hq_ft_layers initialised from ft_layers.')

        # Warm up desired layers
        self.warmup_trainer = None
        if self.is_train:
            warmup_cfg = self.opt['train'].get('warmup', None)
            if warmup_cfg and warmup_cfg.get('modules'):
                self.warmup_trainer = WarmupTrainer(self.net_G, warmup_cfg)
                if self.verbose:
                    self.logger.info('WarmupTrainer initialised.')

        # init ema
        self.init_ema_network()

        # Define and load discriminator net
        if self.is_train and 'gan_crit' in self.opt['train']:
            # define net D
            net_D_opt = self.opt['model']['discriminator']
            net_D_name = self.opt['model']['d_name'].lower()
            self.net_D = define_discriminator(net_D_opt, net_D_name=net_D_name).to(self.device)
            if self.verbose:
                self.logger.info('Discriminator: {}\n'.format(
                    self.opt['model']['d_name']) + self.net_D.__str__())

            # load net D
            load_path_D = self.opt['model']['d_load_path']
            if load_path_D is not None:
                self.load_network(self.net_D, load_path_D, strict_load=False)
                if self.verbose:
                    self.logger.info('Loaded discriminator from: {}'.format(
                        load_path_D))
        else: 
            self.net_D = None

        # Set up Facial component discriminator
        # ── Facial discriminator ────────────────────────────────────────
        if self.is_train and self.opt['train'].get('gan_facial_comp_crit') is not None:
            net_D_opt = self.opt['model']['discriminator_facial']
            net_D_name = self.opt['model']['d_name'].lower()
            self.net_D_roi = define_discriminator(net_D_opt, net_D_name=net_D_name).to(self.device)

            load_path_D_facial = self.opt['model'].get('d_facial_load_path')
            if load_path_D_facial:
                self.load_network(self.net_D_roi, load_path_D_facial, strict_load=False)
                if self.verbose:
                    self.logger.info('Loaded facial discriminator from: {}'.format(
                        load_path_D_facial))
        else:
            self.net_D_roi = None

    def config_training(self):
        self.set_criterion()

        # Mixed precision (AMP) — see BaseModel._init_amp
        self._init_amp()

        # Generator optimizer and scheduler
        # All generator params are registered in the optimizer.
        # WarmupTrainer handles freezing/unfreezing via requires_grad — frozen params
        # are skipped automatically by optimizer.step() when requires_grad=False.
        self.optim_G = optim.AdamW(
            self.net_G.parameters(),
            lr=self.opt['train']['generator']['lr'],
            weight_decay=self.opt['train']['generator'].get('weight_decay', 0),
            betas=(
                self.opt['train']['generator'].get('beta1', 0.9),
                self.opt['train']['generator'].get('beta2', 0.999))
        )

        self.sched_G = define_lr_schedule(
            self.opt['train']['generator'].get('lr_schedule'), self.optim_G)
        
        # Discriminator optimizer, scheduler and training flags
        if self.gan_crit is not None:
            discriminator_opt = self.opt['train']['discriminator']
            self.optim_D = optim.AdamW(
                self.net_D.parameters(),
                lr=discriminator_opt['lr'],
                weight_decay=discriminator_opt.get('weight_decay', 0),
                betas=(
                    discriminator_opt.get('beta1', 0.9),
                    discriminator_opt.get('beta2', 0.999))
            )

            self.sched_D = define_lr_schedule(
                discriminator_opt.get('lr_schedule'), self.optim_D)

            # regularization weights
            gan_crit_opt = self.opt['train']['gan_crit']
            self.r1_reg_weight = gan_crit_opt['r1_reg_weight']  # for discriminator
            self.net_d_iters = gan_crit_opt.get('net_d_iters', 1)
            self.net_d_init_iters = gan_crit_opt.get('net_d_init_iters', 0)
            self.net_d_reg_every = gan_crit_opt['net_d_reg_every'] 
            self.net_d_start_iters = gan_crit_opt['net_d_start_iters'] 
        else:
            self.net_d_iters = 1  # Update G every iteration
            self.net_d_init_iters = 0
            self.net_d_start_iters = 0

        # ── Facial discriminator ────────────────────────────────────────
        if self.net_D_roi is not None:
            facial_d_opt = self.opt['train']['discriminator_facial']
            self.optim_D_facial = optim.AdamW(
                self.net_D_roi.parameters(),
                lr=facial_d_opt['lr'],
                weight_decay=facial_d_opt.get('weight_decay', 0),
                betas=(
                    facial_d_opt.get('beta1', 0.9),
                    facial_d_opt.get('beta2', 0.999))
            )
            self.sched_D_facial = define_lr_schedule(
                facial_d_opt.get('lr_schedule'), self.optim_D_facial)

            gan_facial_opt = self.opt['train']['gan_facial_comp_crit']
            self.r1_reg_weight_facial = gan_facial_opt.get('r1_reg_weight', 0.0)
            self.net_d_facial_reg_every  = gan_facial_opt.get('net_d_reg_every', 16)

    def set_criterion(self):
        train_opt = self.opt['train']

        # Reconstruction criterion
        self.pix_crit = define_criterion(train_opt.get('pixel_crit'))

        # Perceptual crit
        self.feat_crit = define_criterion(train_opt.get('feature_crit'))
        if self.feat_crit is not None:
            self.feat_crit = self.feat_crit.to(self.device)

        # Identity criterion
        self.id_crit = define_criterion(train_opt.get('id_crit'))
        if self.id_crit is not None:
            # Load id crit configuration
            id_crit_dict = train_opt['id_crit']
            id_net_path = id_crit_dict['id_network_path']

            # Define and load ID net
            if not hasattr(self, 'net_ID'):
                self.net_ID = define_id_net(id_net_path).to(self.device)
            else:
                print("Skipping net_ID initialization: already initialized.")

            # Hard loss flag
            # False - loss computed only between HR reconstruction and GT
            # True - loss compited also between HR reconstruction and Reference(s)
            self.id_crit_hard = id_crit_dict.get('hard_loss', False)

        # Facial component criterion
        self.facial_comp_crit = define_criterion(train_opt.get('facial_comp_crit'))
        if self.facial_comp_crit is not None:
            facial_comp_dict = train_opt['facial_comp_crit']
            self.facial_comp_crit_hard = facial_comp_dict.get('hard_loss', False)

        # GAN crit
        self.gan_crit = define_criterion(train_opt.get('gan_crit'))

        # ── GAN facial component crit
        self.gan_facial_comp_crit = define_criterion(
            self.opt['train'].get('gan_facial_comp_crit'))

        # Facial components info
        if self.facial_comp_crit is not None or self.gan_facial_comp_crit is not None:
            facial_roi_dict = train_opt['facial_roi']
            # Eye / Mouth enlarge factors
            self.eye_enlarge = facial_roi_dict.get('enlarge_eye', [1.5, 1.5])
            self.mouth_enlarge = facial_roi_dict.get('enlarge_mouth', [1.2, 1.2])

            # Eye / Mouth output sizes
            eye_size_base = facial_roi_dict.get('eye_size', 80)
            mouth_size_base = facial_roi_dict.get('mouth_size', 120)
            face_ratio = facial_roi_dict.get('face_ratio', 1)
            self.eye_out_size = [
                int(eye_size_base * face_ratio * self.eye_enlarge[1]),  # height
                int(eye_size_base * face_ratio * self.eye_enlarge[0])   # width
            ]
            self.mouth_out_size = [
                int(mouth_size_base * face_ratio * self.mouth_enlarge[1]),  # height
                int(mouth_size_base * face_ratio * self.mouth_enlarge[0])   # width
            ]

        # CodeFormer specific losses
        self.hq_feat_loss = train_opt.get('use_hq_feat_loss', True)
        self.feat_loss_weight = train_opt.get('feat_loss_weight', 1.0)
        self.cross_entropy_loss = train_opt.get('cross_entropy_loss', True)
        self.entropy_loss_weight = train_opt.get('entropy_loss_weight', 0.5)
        self.use_adaptive_weight = train_opt.get('use_adaptive_weight', False)
        self.scale_adaptive_gan_weight = train_opt.get('scale_adaptive_gan_weight', 0.8)

        # Generator fix flag
        self.fix_generator   = self.warmup_trainer.is_module_frozen('generator') \
                       if self.warmup_trainer else False
        self.fix_transformer = self.warmup_trainer.is_module_frozen('ft_layers') \
                            if self.warmup_trainer else False
        self.logger.info(f'fix_generator: {self.fix_generator}. fix_transformer: {self.fix_transformer}')

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)

    def feed_data(self, data):
        # --- Basic data ---
        self.lr = data['lr'].to(self.device)
        self.gt = data.get('gt', None)
        if self.gt is not None:
            self.gt = self.gt.to(self.device)

        # --- Reference images ---
        self.ref = data.get('ref', None)
        if isinstance(self.ref, torch.Tensor) and self.ref.numel() > 0:
            self.ref = self.ref.to(self.device)
        else:
            self.ref = None

        # --- ID embeddings ---
        self.id_embeddings = data.get('id_embeddings', None)
        if isinstance(self.id_embeddings, dict):
            for k in self.id_embeddings:
                self.id_embeddings[k] = self.id_embeddings[k].to(self.device)
        
        # --- FaRL token embeddings ---
        self.farl_embeddings = data.get('farl_embeddings', None)
        if isinstance(self.farl_embeddings, dict):
            for k in self.farl_embeddings:
                self.farl_embeddings[k] = self.farl_embeddings[k].to(self.device)

        # --- Online embedding computation (ref_augment + training only) ---
        # Reference images are augmented at load time so offline caches are stale.
        # ArcFace reuses self.net_ID (same pattern as the id_crit block in train()).
        # FaRL uses self.net_FaRL loaded in _load_farl_encoder().
        # At inference is_train=False so the pre-computed cache is used instead.
        if ( self.ref_augment and self.is_train and self.ref is not None ) or \
            ( not self.is_train and self.ref is not None) :
            B, N, C, H, W = self.ref.shape
            ref_flat = self.ref.view(B * N, C, H, W)  # (B*N, 3, H, W)

            # ArcFace — reuse self.net_ID, identical to the pattern in train()
            if self.id_embeddings is None and self.net_ID is not None:
                with torch.no_grad():
                    arc_embs = self.net_ID(ref_flat)              # (B*N, 512)
                arc_embs = arc_embs.view(B, N, 512).unsqueeze(2)  # (B, N, 1, 512)
                self.id_embeddings = {'refs': arc_embs}

            # FaRL — CLIP ViT-B/16 preprocessing (224x224) then full transformer
            # forward keeping all 197 tokens (see farl_wrapper for the shared core).
            if self.net_FaRL is not None and self.farl_embeddings is None:
                with torch.no_grad():
                    clip_imgs = farl_preprocess_neg1to1(ref_flat)                 # (B*N, 3, 224, 224)
                    x = farl_visual_tokens(self.net_FaRL.visual, clip_imgs)       # (B*N, 197, 768)
                farl_embs = x.view(B, N, 197, 768)                               # (B, N, 197, 768)
                self.farl_embeddings = {'refs': farl_embs}

        # --- Facial components ---
        self.facial_components = data.get('facial_components', None)

        # CodeFormer latents
        if 'latent_gt' in data:
            self.idx_gt = data['latent_gt'].to(self.device)
            self.idx_gt = self.idx_gt.view(self.gt.shape[0], -1)
        else:
            self.idx_gt = None

         # Get ref latent and images if existing
        if 'latent_ref' in data:
            self.idx_ref = data['latent_ref'].to(self.device)
            self.idx_ref = self.idx_ref.view(self.gt.shape[0], -1)
        else:
            self.idx_ref = None

    def train(self, data, iter, accum_steps=1, is_last_accum=True):
        """ Mini-batch training with single-frame image SR """
        # Feed data
        self.feed_data(data)

        # Track whether any optimizer stepped this iter (for AMP scaler.update)
        stepped = False

        # --- Optimize G --- #        
        # Freeze Discriminator
        if self.net_D is not None:
            self.set_requires_grad(self.net_D, False)
        if self.net_D_roi is not None:
            self.set_requires_grad(self.net_D_roi, False)

        # Training mode
        self.net_G.train()

        # Setup training dict
        train_dict = {
            'w': self.w_scale,
            'detach_16': True
        }

        # Extract reference features / k,v pairs
        if self.conditional:
            ref_feat_dict = self._extract_reference_features(return_ref_kv=self.return_ref_kv)
            if ref_feat_dict is not None:
                train_dict['x_ref'] = ref_feat_dict

        # Build id_embs / farl_embs for CodeRefFormerV5
        if self.id_embeddings is not None:
            if 'refs' in self.id_embeddings:
                train_dict['id_embs'] = self.id_embeddings['refs'].squeeze(2)  # (B, N, 512) — refs only

        if self.farl_embeddings is not None:
            if 'refs' in self.farl_embeddings:
                train_dict['farl_embs'] = self.farl_embeddings['refs']  # (B, N, 197, 768) — refs only

        # Forward through the network (frozen encoder/generator run in reduced
        # precision under autocast; outputs cast back to fp32 for stable losses).
        with self._amp_autocast():
            self.output, self.logits, self.lq_feat = self.net_G(self.lr, **train_dict)
        if self.use_amp:
            self.output  = self.output.float()
            self.logits  = self.logits.float()
            self.lq_feat = self.lq_feat.float()
        if self.hq_feat_loss:
            # quant_feats
            quant_feat_gt = self.net_G.quantize.get_codebook_feat(self.idx_gt, 
                                                                  shape=[self.gt.shape[0],16,16,256])

        # Save some debugging outputs
        if iter % 2 == 0 and is_last_accum:
            temp = tensor2img(self.lr[0:1], rgb2bgr=True, min_max=(-1, 1))
            cv2.imwrite(f'debug/lr.png', temp)
            temp = tensor2img(self.output[0:1], rgb2bgr=True, min_max=(-1, 1))
            cv2.imwrite(f'debug/sr.png', temp)
            temp = tensor2img(self.gt[0:1], rgb2bgr=True, min_max=(-1, 1))
            cv2.imwrite(f'debug/gt.png', temp)
            if self.ref is not None and self.ref.numel() != 0:
                temp = tensor2img(self.ref[0:1], rgb2bgr=True, min_max=(-1, 1))
                cv2.imwrite(f'debug/ref.png', temp)

        # Extract GT latent if needed
        if self.idx_gt is None:
            # Generate GT latent
            x = self.hq_encoder(self.gt)
            _, _, quant_stats = self.hq_quantize(x)
            min_encoding_indices = quant_stats['min_encoding_indices']
            self.idx_gt = min_encoding_indices.view(self.gt.shape[0], -1)

        # Compute the compound loss
        loss_G = 0
        self.log_dict = OrderedDict()

        if (iter % self.net_d_iters == 0 and iter >= self.net_d_init_iters):
            # CodeFormer Losses
            # hq_feat_loss
            if not self.fix_transformer:
                if self.hq_feat_loss: # codebook loss 
                    l_feat_encoder = torch.mean((quant_feat_gt.detach()-self.lq_feat)**2) * self.feat_loss_weight
                    loss_G += l_feat_encoder
                    self.log_dict['l_feat_encoder'] = l_feat_encoder.item()

            # cross_entropy_loss
            if self.cross_entropy_loss:
                # b(hw)n -> bn(hw)
                cross_entropy_loss = F.cross_entropy(self.logits.permute(0, 2, 1), self.idx_gt) * self.entropy_loss_weight
                loss_G += cross_entropy_loss
                self.log_dict['cross_entropy_loss'] = cross_entropy_loss.item()

            # Note: CodeFormer does not use image losses with high degradations=> let's try and see
            # Reconstruction loss
            if self.pix_crit is not None:
                pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
                loss_pix_G = pix_w * self.pix_crit(self.output, self.gt)
                loss_G += loss_pix_G
                self.log_dict['l_pix_G'] = loss_pix_G.item()

            # Feature (perceptual) loss - Not weighted
            if self.feat_crit is not None:
                loss_feat_G = self.feat_crit(self.output, self.gt)
                feat_w = self.opt['train']['feature_crit'].get('weight', 1)
                loss_feat_G *= feat_w
                loss_G += loss_feat_G
                self.log_dict['l_feat_G'] = loss_feat_G.item()

            # Identity Loss
            if self.id_crit is not None:
                # Compute HR ID embedding
                hr_id_embedding = self.net_ID(self.output)

                # Compute GT ID embeddings
                id_emb = self.id_embeddings
                if id_emb is not None and len(self.id_embeddings) > 0:
                    gt_id_embedding = id_emb['gt']
                else:
                    with torch.no_grad():
                        gt_id_embedding = self.net_ID(self.gt)

                # Compute Refs ID embeddings
                if self.id_crit_hard:
                    if self.ref_augment or 'refs' not in id_emb:  # Compute ID embeddings on augmented images
                        ref_data = self.ref  # [B, N, 3, H, W]
                        B, N, C, H, W = ref_data.shape

                        ref_data = ref_data.view(B * N, C, H, W)

                        with torch.no_grad():
                            ref_id_embedding = self.net_ID(ref_data)  # [B*N, 512]
                        ref_id_embedding = ref_id_embedding.view(B, N, -1)  # [B, N, 512]
                        ref_id_embedding = ref_id_embedding.mean(dim=1)     # [B, 512]
                    else:
                        ref_id_embedding = torch.mean(id_emb['refs'], dim=1)

                # Compute the ID loss
                hr_id_embedding = hr_id_embedding.to(self.device).squeeze(1)
                gt_id_embedding = gt_id_embedding.to(self.device).squeeze(1)
                # hr_id_embedding = hr_id_embedding.to(self.device)
                # gt_id_embedding = gt_id_embedding.to(self.device)
                # hr_id_embedding = hr_id_embedding.unsqueeze(1)
                # print(hr_id_embedding.shape)
                # print(gt_id_embedding.shape)
                # print(ref_id_embedding.shape)

                loss_id_G = self.id_crit(hr_id_embedding, gt_id_embedding)
                if self.id_crit_hard:
                    # ref_id_embedding = ref_id_embedding.to(self.device).squeeze(1)
                    ref_id_embedding = ref_id_embedding.to(self.device)
                    loss_id_G += self.id_crit(hr_id_embedding, ref_id_embedding)
                loss_id_G *= self.opt['train']['id_crit'].get('weight', 1)
                loss_G += loss_id_G
                self.log_dict['l_id_G'] = loss_id_G.item()

            # Facial component loss
            if self.facial_comp_crit is not None:
                # Get facial component regions (ROIs)               
                facial_components = self.get_roi_regions(self.output, self.gt, 
                                                         ref_data=data.get('ref', None),
                                                         components=data['facial_components'])
                
                # Compute loss
                loss_facial_comp = 0
                for comp in ['left_eye', 'right_eye', 'mouth']:
                    hr_comp = facial_components['hr'][comp]
                    gt_comp = facial_components['gt'][comp]

                    comp_loss = self.facial_comp_crit(hr_comp, gt_comp)
                    loss_facial_comp += comp_loss
                    self.log_dict[f'l_facial_{comp}_gt_G'] = comp_loss.item()
                    if self.facial_comp_crit_hard and facial_components['ref'] is not None:
                        ref_comp = facial_components['ref'][comp]
                        comp_loss_ref = torch.stack([
                            self.facial_comp_crit(hr_comp, r)
                            for r in ref_comp[comp]
                        ]).mean()
                        loss_facial_comp += comp_loss_ref
                        self.log_dict[f'l_facial_{comp}_ref_G'] = comp_loss_ref.item()                  
                loss_facial_comp *= self.opt['train']['facial_comp_crit'].get('weight', 1)
                loss_G += loss_facial_comp
                self.log_dict['l_facial_comp_G'] = loss_facial_comp.item()

            # GAN Loss
            if self.gan_crit is not None:
                # Not using adaptive weight right now
                fake_g_pred = self.net_D(self.output)
                loss_gan_G = self.gan_crit(fake_g_pred, True, is_critic_update=False)
                if self.use_adaptive_weight:
                    reconstruct_loss = loss_pix_G + loss_feat_G
                    if not self.fix_generator:
                        last_layer = self.net_G.generator.blocks[-1].weight
                    else:
                        largest_fuse_size = self.opt['model']['generator']['connect_list'][-1]
                        last_layer = self.net_G.fuse_convs_dict[largest_fuse_size].shift[-1].weight
                        
                    # Check if last layer requires grad -> can be frozen during warmup phase
                    if last_layer.requires_grad:
                        d_weight = self._calculate_adaptive_weight(
                            reconstruct_loss, loss_gan_G, last_layer, disc_weight_max=1.0)
                        d_weight *= self.scale_adaptive_gan_weight
                        self.log_dict['d_weight'] = d_weight
                        loss_gan_G *= d_weight
                    else:
                        loss_gan_G *= self.opt['train']['gan_crit'].get('weight', 1)
                else:
                    loss_gan_G *= self.opt['train']['gan_crit'].get('weight', 1)
                
                loss_G += loss_gan_G
                self.log_dict['l_gan_G'] = loss_gan_G.item()

            # Facial component GAN loss
            if self.net_D_roi is not None and self.gan_facial_comp_crit is not None:
                # Refolosim roi_regions deja calculate mai sus (dacă facial_comp_crit
                # e activ); altfel le calculăm acum.
                if 'facial_components' not in locals():
                    facial_components = self.get_roi_regions(
                        self.output, self.gt,
                        ref_data=data.get('ref', None),
                        components=data['facial_components']
                    )

                # Concat all components per batch: [N_comp * B, C, H, W]
                hr_comps = torch.cat([
                    facial_components['hr']['left_eye'],   # [B, C, He, We]
                    facial_components['hr']['right_eye'],  # [B, C, He, We]
                    F.interpolate(                         # resize mouth → eye size
                        facial_components['hr']['mouth'],  # [B, C, Hm, Wm]
                        size=self.eye_out_size,
                        mode='bilinear',
                        align_corners=False
                    )
                ], dim=1)  # [B, 3*C, He, We]

                gt_comps = torch.cat([
                    facial_components['gt']['left_eye'],
                    facial_components['gt']['right_eye'],
                    F.interpolate(
                        facial_components['gt']['mouth'],
                        size=self.eye_out_size,
                        mode='bilinear',
                        align_corners=False
                    )
                ], dim=1)  # [B, 3*C, He, We]

                # G update: freeze D_facial
                self.set_requires_grad(self.net_D_roi, False)
                fake_facial_pred = self.net_D_roi(hr_comps)
                loss_gan_facial_G = self.gan_facial_comp_crit(
                    fake_facial_pred, True, is_critic_update=False
                )
                loss_gan_facial_G *= self.opt['train']['gan_facial_comp_crit'].get('weight', 1)
                loss_G += loss_gan_facial_G
                self.log_dict['l_gan_facial_comp_G'] = loss_gan_facial_G.item()

            # Optimize G
            loss_G = loss_G / accum_steps
            self.amp_scaler.scale(loss_G).backward()

            # Step after accum_steps iterations
            if is_last_accum:
                self.amp_scaler.step(self.optim_G)
                self.optim_G.zero_grad()
                stepped = True

                # update ema network
                if self.ema_decay > 0:
                    self.model_ema(decay=self.ema_decay)

        # --- Optimize D --- #
        # They also do not optimize D on hard degradations
        if self.net_D is not None and iter >= self.net_d_start_iters:
            # Unfreeze Discriminator
            self.set_requires_grad(self.net_D, True)
            if is_last_accum:
                self.optim_D.zero_grad()

            # Predict
            fake_d_pred = self.net_D(self.output.detach())
            real_d_pred = self.net_D(self.gt)

            # Compute GAN loss
            loss_gan_D = self.gan_crit(fake_d_pred, False, is_critic_update=True) + \
                self.gan_crit(real_d_pred, True, is_critic_update=True)
        
            loss_gan_D = loss_gan_D / accum_steps
            self.amp_scaler.scale(loss_gan_D).backward()

            self.log_dict['l_gan_D'] = loss_gan_D.item()
            self.log_dict['real_score'] = real_d_pred.detach().mean().item()  # Also log real/fake logits
            self.log_dict['fake_score'] = fake_d_pred.detach().mean().item()

            # R1 lazy regularization term -> every `net_d_reg_evey` iterations
            # (kept in fp32 — gradient penalty is unstable in low precision)
            if self.r1_reg_weight > 0 and is_last_accum and iter % self.net_d_reg_every == 0:
                self.gt.requires_grad_(True)
                real_pred = self.net_D(self.gt)
                r1_term = r1_penalty(real_pred, self.gt)
                r1_term = (r1_term * self.r1_reg_weight * 0.5 * self.net_d_reg_every)
                self.amp_scaler.scale(r1_term).backward()
                self.gt.requires_grad_(False)
                self.log_dict['r1_term_D'] = r1_term.detach().mean().item()

            if is_last_accum:
                self.amp_scaler.step(self.optim_D)
                stepped = True
            if self.gt.grad is not None:
                self.gt.grad = None

        # ── Optimize D facial ───────────────────────────────────────────
        if self.net_D_roi is not None and iter >= self.net_d_start_iters:
            self.set_requires_grad(self.net_D_roi, True)
            if is_last_accum:
                self.optim_D_facial.zero_grad()

            # hr_comps / gt_comps sunt deja calculate mai sus în același iter,
            # dar hr_comps trebuie detached pentru update-ul D.
            fake_facial_pred_d = self.net_D_roi(hr_comps.detach())
            real_facial_pred_d = self.net_D_roi(gt_comps)

            loss_gan_facial_D = (
                self.gan_facial_comp_crit(fake_facial_pred_d, False, is_critic_update=True) +
                self.gan_facial_comp_crit(real_facial_pred_d, True,  is_critic_update=True)
            )

            loss_gan_facial_D = loss_gan_facial_D / accum_steps
            self.amp_scaler.scale(loss_gan_facial_D).backward()

            self.log_dict['l_gan_facial_comp_D']      = loss_gan_facial_D.item()
            self.log_dict['facial_comp_real_score']   = real_facial_pred_d.detach().mean().item()
            self.log_dict['facial_comp_fake_score']   = fake_facial_pred_d.detach().mean().item()

            # ── R1 regularization (opțional din config) ──────────────────
            if self.r1_reg_weight_facial > 0 and is_last_accum and iter % self.net_d_facial_reg_every == 0:
                gt_comps.requires_grad_(True)
                real_pred_r1 = self.net_D_roi(gt_comps)
                r1_term_facial = r1_penalty(real_pred_r1, gt_comps)
                r1_term_facial = (
                    r1_term_facial
                    * self.r1_reg_weight_facial
                    * 0.5
                    * self.net_d_facial_reg_every
                )
                self.amp_scaler.scale(r1_term_facial).backward()
                gt_comps.requires_grad_(False)
                if gt_comps.grad is not None:
                    gt_comps.grad = None
                self.log_dict['r1_term_D_facial'] = r1_term_facial.detach().mean().item()

            if is_last_accum:
                self.amp_scaler.step(self.optim_D_facial)
                stepped = True

        # Update iteration index
        if is_last_accum:
            # Update the AMP grad scaler once per iter, only if an optimizer
            # stepped (no-op for bf16 / disabled scaler).
            if stepped:
                self.amp_scaler.update()

            # Unfreeze any warm-up layers if warm-up iterations are over
            if self.warmup_trainer is not None:
                self.warmup_trainer.step(iter)

    def infer(self, data, **kwargs):
        """ Inference for a single low-resolution image (3D tensor HWC or CHW) """
        # Feed data
        self.feed_data(data)

        # Get references if provided
        # Get condition => if missing, generate with denoising UNet
        infer_dict = {
            'w': self.w_scale
        }

        # Extract reference features
        if self.conditional:
            ref_feat_dict = self._extract_reference_features(return_ref_kv=self.return_ref_kv)
            if ref_feat_dict is not None:
                infer_dict['x_ref'] = ref_feat_dict

        # Build id_embs / farl_embs for CodeRefFormerV5
        if self.id_embeddings is not None:
            if 'refs' in self.id_embeddings:
                infer_dict['id_embs'] = self.id_embeddings['refs'].squeeze(2)  # (B, N, 512) — refs only

        if self.farl_embeddings is not None:
            if 'refs' in self.farl_embeddings:
                infer_dict['farl_embs'] = self.farl_embeddings['refs']  # (B, N, 197, 768) — refs only

        # Forward through the network
        # Get network 
        infer_net = self.net_G_ema if self.ema_decay > 0 else self.net_G

        # Perform inference
        with torch.no_grad():
            sr_images, _, _ = infer_net(self.lr, **infer_dict)
        sr_images = sr_images.cpu()

        return_dict = {
            'hr_data': sr_images
        }
        return return_dict

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
        if self.net_D is not None:
            self.save_network(self.net_D, 'D', current_iter)
        if self.net_D_roi is not None:                          
            self.save_network(self.net_D_roi, 'D_roi', current_iter)
    
    def build_rois_from_components(self, components, device, batch_idx):
        rois_eyes = []
        rois_mouths = []

        def make_box(cx, cy, r, ex, ey):
            half_w = r * ex
            half_h = r * ey
            return torch.tensor(
                [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
                device=device
            )

        # left / right eye
        for key in ['left_eye', 'right_eye']:
            cx = components[key][0][batch_idx]  # x
            cy = components[key][1][batch_idx]  # y
            r  = components[key][2][batch_idx]  # r
            # print(f"r={r} for {key} at batch idx {batch_idx}.")
            rois_eyes.append(
                torch.cat([torch.tensor([batch_idx], device=device),
                        make_box(cx, cy, r, self.eye_enlarge[0], self.eye_enlarge[1])
                ])
            )
        # print(f"Built ROIs for eyes: {rois_eyes}.")

        # mouth
        cx = components['mouth'][0][batch_idx]  # x
        cy = components['mouth'][1][batch_idx]  # y
        r  = components['mouth'][2][batch_idx]  # r
        # print(f"r={r} for mouth at batch idx {batch_idx}.")
        rois_mouths.append(
            torch.cat([torch.tensor([batch_idx], device=device),
                    make_box(cx, cy, r, self.mouth_enlarge[0], self.mouth_enlarge[1])
            ])
        )
        # print(f"Built ROIs for mouth: {rois_mouths}.")

        return rois_eyes, rois_mouths
    
    def get_roi_regions(
        self,
        hr_data, gt_data,
        ref_data=None,
        components=None
    ):
        device = hr_data.device
        B = gt_data.size(0)

        # ===================== GT / HR =====================
        gt_eye_rois, gt_mouth_rois = [], []

        for b in range(B):
            comp = components['gt']
            if comp is None:
                continue

            e, m = self.build_rois_from_components(comp, device, b)
            gt_eye_rois.extend(e)
            gt_mouth_rois.extend(m)

        gt_eye_rois = torch.stack(gt_eye_rois).to(device).float()
        gt_mouth_rois = torch.stack(gt_mouth_rois).to(device).float()

        # GT
        gt_eyes = roi_align(gt_data, gt_eye_rois, self.eye_out_size)
        gt_mouths = roi_align(gt_data, gt_mouth_rois, self.mouth_out_size)

        # HR (same boxes as GT)
        hr_eyes = roi_align(hr_data, gt_eye_rois, self.eye_out_size)
        hr_mouths = roi_align(hr_data, gt_mouth_rois, self.mouth_out_size)

        gt_left_eye, gt_right_eye = gt_eyes[0::2], gt_eyes[1::2]
        hr_left_eye, hr_right_eye = hr_eyes[0::2], hr_eyes[1::2]

        # ===================== REFERENCES =====================
        ref_out = None
        if ref_data is not None and hasattr(self, 'facial_comp_crit_hard') and 'refs' in components:
            ref_left, ref_right, ref_mouth = [], [], []

            for k, ref_comp in enumerate(components['refs']):
                if ref_comp is None:
                    continue

                e, m = self.build_rois_from_components(ref_comp, device, k)

                e = torch.stack(e)
                m = torch.stack(m)

                ref_img = ref_data[:, k]  # [B,3,H,W]

                eyes = roi_align(ref_img, e, self.eye_out_size)
                mouths = roi_align(ref_img, m, self.mouth_out_size)

                ref_left.append(eyes[0::2])
                ref_right.append(eyes[1::2])
                ref_mouth.append(mouths)

            ref_out = {
                'left_eye': ref_left,
                'right_eye': ref_right,
                'mouth': ref_mouth
            }

        return {
            'gt': {
                'left_eye': gt_left_eye,
                'right_eye': gt_right_eye,
                'mouth': gt_mouths
            },
            'hr': {
                'left_eye': hr_left_eye,
                'right_eye': hr_right_eye,
                'mouth': hr_mouths
            },
            'ref': ref_out
        }