from collections import OrderedDict
import os.path as osp

import torch

from utils.base_utils import get_logger
from models.networks import define_generator


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.verbose = opt['verbose']
        self.scale = opt['scale']
        self.logger = get_logger('base')
        self.device = torch.device(opt['device'])
        self.is_train = opt['is_train']

        # Mixed-precision defaults (overridden by _init_amp() at train time).
        self.use_amp = False
        self.amp_dtype = torch.float16
        self.amp_scaler = torch.amp.GradScaler('cuda', enabled=False)

        if self.is_train:
            self.ckpt_dir = opt['train']['ckpt_dir']
            self.log_decay = opt['logger'].get('decay', 0.99)
            self.log_dict = OrderedDict()
            self.running_log_dict = OrderedDict()

    def _init_amp(self):
        """Configure mixed-precision training from opt['train']['amp'].

        amp:
          enabled: bool   — turn AMP on/off (default off → full fp32).
          dtype:   'bf16' | 'fp16'  — autocast compute dtype (default bf16).

        bf16 is recommended (and default): it has the same exponent range as
        fp32, so no gradient scaling is needed and it is stable for GAN losses,
        r1 penalty and the adaptive weight. fp16 additionally uses a GradScaler.
        Frozen sub-networks (HQ encoder, generator, ArcFace/FaRL, flow net) run
        in the reduced precision under autocast, saving memory and time.
        """
        amp_opt = self.opt['train'].get('amp', {}) or {}
        self.use_amp = bool(amp_opt.get('enabled', False))
        dtype_str = str(amp_opt.get('dtype', 'bf16')).lower()
        self.amp_dtype = torch.bfloat16 if dtype_str in ('bf16', 'bfloat16') else torch.float16
        # GradScaler only needed for fp16 (bf16 keeps the fp32 exponent range).
        self.amp_scaler = torch.amp.GradScaler(
            'cuda', enabled=(self.use_amp and self.amp_dtype == torch.float16))
        if self.use_amp and self.verbose:
            self.logger.info(
                f'[AMP] enabled — dtype={dtype_str}, '
                f'grad_scaler={self.amp_scaler.is_enabled()}')

    def _amp_autocast(self):
        """Autocast context for forward passes (no-op when AMP is disabled)."""
        return torch.autocast(device_type='cuda', dtype=self.amp_dtype,
                              enabled=self.use_amp)

    def set_network(self):
        pass

    def config_training(self):
        pass

    def set_criterion(self):
        pass

    def train(self, data):
        pass

    def infer(self, data):
        pass

    def update_learning_rate(self):
        if hasattr(self, 'sched_G') and self.sched_G is not None:
            self.sched_G.step()

        if hasattr(self, 'sched_D') and self.sched_D is not None:
            self.sched_D.step()

    def get_current_learning_rate(self):
        lr_dict = OrderedDict()

        if hasattr(self, 'optim_G'):
            lr_dict['lr_G'] = self.optim_G.param_groups[0]['lr']

        if hasattr(self, 'optim_D'):
            lr_dict['lr_D'] = self.optim_D.param_groups[0]['lr']

        return lr_dict

    def update_running_log(self):
        d = self.log_decay
        for k in self.log_dict.keys():
            current_val = self.log_dict[k]
            if isinstance(current_val, torch.Tensor):
                current_val = current_val.item()

            running_val = self.running_log_dict.get(k)

            if running_val is None:
                running_val = current_val
            else:
                running_val = d * running_val + (1.0 - d) * current_val

            self.running_log_dict[k] = running_val

    def get_current_log(self):
        return self.log_dict

    def get_running_log(self):
        return self.running_log_dict

    def save(self, current_iter):
        pass

    def init_ema_network(self):
        self.ema_decay = self.opt['model']['generator'].get('ema_decay', 0)
        if self.ema_decay > 0:
            self.logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')

            # define network net_G with Exponential Moving Average (EMA)
            # net_G_ema is used only for testing on one GPU and saving
            self.net_G_ema = define_generator(self.opt).to(self.device)

            # load pretrained model
            load_path_G = self.opt['model']['generator'].get('load_path')
            if load_path_G is not None:
                self.load_network(self.net_G_ema, load_path_G, 
                              strict_load=self.opt['model']['generator'].get('strict_load', True))
                if self.verbose:
                    self.logger.info('Loaded ema generator from: {}'.format(load_path_G))
            else:
                self.model_ema(0)  # copy net_G weight
            self.net_G_ema.eval()

    def save_network(self, net, net_label, current_iter):
        save_filename = '{}_iter{}.pth'.format(net_label, current_iter)
        save_path = osp.join(self.ckpt_dir, save_filename)

        if net_label == 'D' or self.ema_decay == 0:
            # Discriminator or Generator and no ema weights
            torch.save(net.state_dict(), save_path)
        else:
            # Generator with ema weights
            torch.save(
                {
                    'params': net.state_dict(),
                    'params_ema': self.net_G_ema.state_dict()
                },
                save_path
            )

    def save_training_state(self, current_epoch, current_iter):
        # TODO
        pass

    def load_network(self, net, load_path, strict_load=True):
        ckpt = torch.load(load_path)
        if 'params_ema' in ckpt:
            ckpt = ckpt['params_ema']  # TODO: undo
        elif 'params' in ckpt:
            ckpt = ckpt['params']
        # ckpt = ckpt['params']
        
        # Load state dict
        load_result = net.load_state_dict(ckpt, 
                                          strict=strict_load)
        
        # Print out any missing items
        if not strict_load:
            print(">> Missing keys:")
            for key in load_result.missing_keys:
                print(f"  {key}")

            print("\n>> Unexpected keys:")
            for key in load_result.unexpected_keys:
                print(f"  {key}")


    def load_reconstruction_block(self, net, reconstruction_path):
        # Load the pretrained block:
        print(f"Loading reconstruction module: {reconstruction_path}.")
        pretrained_dict = torch.load(reconstruction_path)

        # Get the current model's state_dict
        model_dict = net.state_dict()

        # Check if conv_in weights are compatible
        conv_in_key = 'conv_in.0.weight'  # The first conv layer in conv_in
        if conv_in_key in pretrained_dict:
            pretrained_conv_in = pretrained_dict[conv_in_key]
            # current_conv_in = model_dict[f'srnet.{conv_in_key}']
            
            # Check shape compatibility
            print("Checking reconstruction module compatibility...")
            if pretrained_conv_in.shape != net.reconstruction_channels:
                # print(f'[WARNING] conv_in shape mismatch: {pretrained_conv_in.shape} vs {current_conv_in.shape}')
                print('-> Skipping loading of conv_in weights.')
                # Remove all keys from conv_in
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('conv_in.')}
        else:
            print('[WARNING] conv_in weights not found in checkpoint.')

        # Filter the pretrained dict to match keys in the current model
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # Update model state and load
        model_dict.update(filtered_dict)
        net.load_state_dict(model_dict)

    def pad_sequence(self, lr_data):
        """
        Parameters:
            :param lr_data: tensor in shape tchw
        """
        padding_mode = self.opt['test'].get('padding_mode', 'reflect')
        n_pad_front = self.opt['test'].get('num_pad_front', 0)

        if padding_mode == 'reflect':
            lr_data = torch.cat(
                [lr_data[1: 1 + n_pad_front, ...].flip(0), lr_data], dim=0)

        elif padding_mode == 'replicate':
            lr_data = torch.cat(
                [lr_data[:1, ...].expand(n_pad_front, -1, -1, -1), lr_data],
                dim=0)

        elif padding_mode == 'dual-reflect':
            lr_data = torch.cat(
                [lr_data[1: 1+n_pad_front, ...].flip(0), lr_data, lr_data[-1-n_pad_front: -1, ...].flip(0)],
                dim=0)

        else:
            raise ValueError('Unrecognized padding mode: {}'.format(
                padding_mode))

        return lr_data, n_pad_front

    def set_requires_grad(self, model, flag):
        for p in model.parameters():
            p.requires_grad = flag

    def model_ema(self, decay=0.999):
        net_g_params = dict(self.net_G.named_parameters())
        net_g_ema_params = dict(self.net_G_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    # Freeze some layes
    def freeze_all_except(self, net, layers=[]):
        for name, param in net.named_parameters():
            if any(n in name for n in layers):
                print(f"Training layer: {name}...")
                param.requires_grad = True
            else:
                print(f"Freeze layer: {name}...")
                param.requires_grad = False

    # Unfreeze everything except specific layers
    def unfreeze_all_except(self, net, layers=[]):
        """
        Unfreezes all parameters except those whose names contain any of the strings in 'name'.
        """
        for name, param in net.named_parameters():
            print(name)
            if any(n in name for n in layers):
                param.requires_grad = False
                print(f"Kept frozen: {name}")
            else:
                print(f"Training layer: {name}...")
                param.requires_grad = True

    def _load_farl_encoder(self):
        """Load net_FaRL (CLIP ViT-B/16 with FaRL weights) for on-the-fly token
        extraction during augmented training.

        Path is read from opt['model']['generator']['farl_path'].
        If the key is absent or None, net_FaRL stays None and FaRL tokens will
        not be computed online (use_farl should also be False in that case).

        The network is frozen and set to eval mode, matching the offline
        behaviour of extract_cache.py.
        """
        farl_path = self.opt['model']['generator'].get('farl_path', None)
        if not farl_path:
            if self.verbose:
                self.logger.info('[ref_augment] farl_path not set — net_FaRL not loaded.')
            return

        try:
            import clip
        except ImportError:
            raise ImportError(
                'openai/CLIP is required for online FaRL extraction.\n'
                'Install: pip install git+https://github.com/openai/CLIP.git'
            )

        net_FaRL, _ = clip.load('ViT-B/16', device='cpu')
        state = torch.load(farl_path, map_location='cpu')
        net_FaRL.load_state_dict(state['state_dict'], strict=False)
        self.net_FaRL = net_FaRL.to(self.device)
        self.net_FaRL.eval()
        for p in self.net_FaRL.parameters():
            p.requires_grad = False

        if self.verbose:
            self.logger.info(f'[ref_augment] net_FaRL loaded from: {farl_path}')

    def _extract_reference_features(self, return_ref_kv=False):
        """Encode reference images (self.ref) through the frozen HQ encoder.

        Shared by the image and video models. Returns either multi-scale
        feature maps or per-layer K/V, each as {res: tensor(B, N, ...)}.
        """
        if self.ref is None:
            return None

        B, N, C, H, W = self.ref.shape
        if not self.sequential_refs:
            # --- Batched mode (default): all N references in one forward pass ---
            ref_flat = self.ref.view(B * N, C, H, W)
            with torch.no_grad():
                if return_ref_kv:
                    ref_kv_dict_flat = self.hq_encoder.extract_kv(ref_flat)
                else:
                    _, ref_feat_dict_flat = self.hq_encoder(ref_flat, return_feats=True)

            if return_ref_kv:
                # Reshape K/V to (B, N, C_k, H_k, W_k)
                kv_dict = {}
                for res, (K, V) in ref_kv_dict_flat.items():
                    _, C_k, H_k, W_k = K.shape
                    K = K.view(B, N, C_k, H_k, W_k)
                    V = V.view(B, N, C_k, H_k, W_k)
                    kv_dict[res] = (K, V)
                return kv_dict
            else:
                # Reshape feats to (B, N, C_i, H_i, W_i)
                ref_feat_dict = {}
                for res, feat in ref_feat_dict_flat.items():
                    _, C_i, H_i, W_i = feat.shape
                    ref_feat_dict[res] = feat.view(B, N, C_i, H_i, W_i)
                return ref_feat_dict

        else:
            # --- Sequential mode (sequential_refs: true): one reference at a time ---
            # Lower peak VRAM — activations for only one ref exist at a time.
            # Results are identical to batched mode.
            if return_ref_kv:
                kv_accum = {}  # res -> (list[K], list[V])
                for n in range(N):
                    with torch.no_grad():
                        kv_dict_n = self.hq_encoder.extract_kv(self.ref[:, n])
                    for res, (K, V) in kv_dict_n.items():
                        if res not in kv_accum:
                            kv_accum[res] = ([], [])
                        kv_accum[res][0].append(K.unsqueeze(1))  # (B, 1, C_k, H_k, W_k)
                        kv_accum[res][1].append(V.unsqueeze(1))
                return {
                    res: (torch.cat(ks, dim=1), torch.cat(vs, dim=1))
                    for res, (ks, vs) in kv_accum.items()
                }
            else:
                feat_accum = {}  # res -> list of (B, 1, C_i, H_i, W_i)
                for n in range(N):
                    with torch.no_grad():
                        _, feat_dict_n = self.hq_encoder(self.ref[:, n], return_feats=True)
                    for res, feat in feat_dict_n.items():
                        if res not in feat_accum:
                            feat_accum[res] = []
                        feat_accum[res].append(feat.unsqueeze(1))
                return {
                    res: torch.cat(feats, dim=1)  # (B, N, C_i, H_i, W_i)
                    for res, feats in feat_accum.items()
                }

    def _calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        return d_weight