import cv2
import json
import numpy as np
import os
import os.path as osp
import pyiqa
import torch

from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
from models.networks.face_descriptor_nets.arcface_wrapper import IdentityEncoder
from utils import base_utils, data_utils, net_utils
from .LPIPS.models.dist_model import DistModel
from torchvision.transforms.functional import normalize


class MetricCalculator():
    """ Metric calculator for model evaluation

        Currently supported metrics:
            * PSNR  (RGB and Y channel)
            * SSIM  (Y channel)
            * LPIPS
            * IDS   (identity cosine similarity vs GT)
            * sigma_ids (std of IDS across a video sequence — video only)
            * IDSRef    (to be implemented)
            * NIQE  (no-reference)
            * MUSIQ (no-reference)
            * CLIPIQA (no-reference)
            * tOF   (temporal optical-flow error — video only)
            * EWarp (estimated warp error — video only)
    """

    def __init__(self, opt):
        # initialize
        self.metric_opt = opt['metric']
        self.device = torch.device(opt['device'])
        self.gt_bit_depth = opt['gt_bit_depth']
        self.psnr_colorspace = ''
        self.dm = None
        self.metric_dict = OrderedDict()
        self.is_video = True

        # ── Per-metric initialization ─────────────────────────────────────────
        for metric_type, cfg in self.metric_opt.items():
            if metric_type.lower() == 'psnr':
                self.psnr_mult        = cfg.get('mult', 1)
                self.psnr_colorspace  = cfg['colorspace']

            elif metric_type.lower() == 'ssim':
                self.ssim_mult = cfg.get('mult', 1)

            elif metric_type.lower() == 'lpips':
                self.lpips_mult = cfg.get('mult', 1)
                self.dm = DistModel()
                self.dm.initialize(
                    model=cfg['model'],
                    net=cfg['net'],
                    colorspace=cfg['colorspace'],
                    spatial=cfg['spatial'],
                    use_gpu=(opt['device'] == 'cuda'),
                    gpu_ids=[0],
                    version=cfg['version'])

            elif metric_type.lower() == 'ids':
                self.ids_mult         = cfg.get('mult', 1)
                self.ids_network_path = cfg['id_network_path']

            # FIX 1: sigma_ids was completely missing from __init__
            elif metric_type.lower() == 'sigma_ids':
                self.sigma_ids_mult   = cfg.get('mult', 1)
                self.ids_network_path = cfg['id_network_path']

            elif metric_type.lower() == 'idsref':
                self.idsref_mult      = cfg.get('mult', 1)
                self.ids_network_path = cfg['id_network_path']

            elif metric_type.lower() == 'niqe':
                self.niqe_mult  = cfg.get('mult', 1)
                self.niqe_model = pyiqa.create_metric('niqe').to(self.device)

            elif metric_type.lower() == 'musiq':
                self.musiq_mult  = cfg.get('mult', 1)
                self.musiq_model = pyiqa.create_metric('musiq').to(self.device)

            elif metric_type.lower() == 'clipiqa':
                self.clipiqa_mult  = cfg.get('mult', 1)
                self.clipiqa_model = pyiqa.create_metric('clipiqa').to(self.device)

            elif metric_type.lower() == 'tof':
                self.tof_mult = cfg.get('mult', 1)

            # FIX 2: ewarp was completely missing from __init__
            elif metric_type.lower() == 'ewarp':
                self.ewarp_mult = cfg.get('mult', 1)

        # FIX 3: original had a typo 'IDSRef`' (backtick) so IDSRef was never
        #        detected; sigma_ids also needs net_ID but was not listed here.
        if any(k in self.metric_opt for k in ('IDS', 'IDSRef', 'sigma_ids')):
            self.net_ID = IdentityEncoder(
                model_path=self.ids_network_path).to(self.device)

        # FIX 4: guard against sigma_ids being requested without IDS, because
        #        sigma_ids is derived from the per-frame IDS values.
        if 'sigma_ids' in self.metric_opt and 'IDS' not in self.metric_opt:
            raise ValueError(
                "sigma_ids requires IDS to also be enabled in metric_opt "
                "— sigma_ids is computed from the per-frame IDS values.")

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def reset(self):
        self.reset_per_sequence()
        self.metric_dict.clear()

    def reset_per_sequence(self):
        self.seq_idx_curr = ''
        self.true_img_cur = None
        self.pred_img_cur = None
        self.true_img_pre = None
        self.pred_img_pre = None

    # ── Aggregation helpers ───────────────────────────────────────────────────

    def get_averaged_results(self):
        metric_avg_dict = {}

        for metric_type in self.metric_opt.keys():
            metric_avg_per_seq = []
            for seq, metric_dict_per_seq in self.metric_dict.items():
                metric_avg_per_seq.append(
                    np.mean(metric_dict_per_seq[metric_type]))

            metric_avg_dict[metric_type] = np.mean(metric_avg_per_seq)

        return metric_avg_dict

    def display_results(self, iter=0, tb_logger=None, ds_name=None):
        logger = base_utils.get_logger('base')

        # Per-sequence results
        for seq, metric_dict_per_seq in self.metric_dict.items():
            logger.info('Sequence: {}'.format(seq))
            for metric_type in self.metric_opt.keys():
                mult = getattr(self, '{}_mult'.format(metric_type.lower()))
                logger.info('\t{}: {:.6f} (x{})'.format(
                    metric_type,
                    mult * np.mean(metric_dict_per_seq[metric_type]),
                    mult))

        # Average across all sequences
        logger.info('Average')
        metric_avg_dict = self.get_averaged_results()
        for metric_type, avg_result in metric_avg_dict.items():
            mult = getattr(self, '{}_mult'.format(metric_type.lower()))
            logger.info('\t{}: {:.6f} (x{})'.format(
                metric_type, mult * avg_result, mult))

        # Log to TensorBoard if a writer is provided
        if tb_logger is not None:
            for key, value in metric_avg_dict.items():
                if ds_name is None:
                    tb_logger.add_scalar(f'val/{key}', value, iter)
                else:
                    tb_logger.add_scalar(f'val/{ds_name}/{key}', value, iter)

    def save_results(self, model_idx, save_path, override=False,
                     iter=None, tb_logger=None):
        # Load existing results file if it already exists
        if osp.exists(save_path):
            with open(save_path, 'r') as f:
                json_dict = json.load(f)
        else:
            json_dict = dict()

        if model_idx not in json_dict:
            json_dict[model_idx] = dict()

        metric_avg_dict = self.get_averaged_results()
        for metric_type, avg_result in metric_avg_dict.items():
            # Skip if metric already recorded and override is disabled
            if metric_type in json_dict[model_idx] and not override:
                continue
            json_dict[model_idx][metric_type] = '{:.6f}'.format(avg_result)

        json_dict = OrderedDict(sorted(json_dict.items()))

        with open(save_path, 'w') as f:
            json.dump(json_dict, f, sort_keys=False, indent=4)

    # ── Dataset / sequence entry points ──────────────────────────────────────

    def compute_dataset_metrics(self, true_dir, pred_dir, sequence_list=None):
        """ Compute metrics for a whole dataset whose root contains one
            sub-folder per video clip / sequence. """

        if sequence_list is None:
            sequence_list = sorted(list(
                set(os.listdir(true_dir)) & set(os.listdir(pred_dir))))

        for seq in sequence_list:
            true_seq_dir = osp.join(true_dir, seq)
            pred_seq_dir = osp.join(pred_dir, seq)
            self.compute_sequence_metrics(seq, true_seq_dir, pred_seq_dir)

    def compute_sequence_metrics(self, seq, true_seq_dir, pred_seq_dir,
                                 true_seq=None, pred_seq=None):
        """ Compute metrics for a sequence or a single image.

        Args:
            seq          : sequence identifier string.
            true_seq_dir : GT directory (ignored when true_seq is given).
            pred_seq_dir : prediction directory (ignored when pred_seq is given).
            true_seq     : numpy array — (H, W, C) for a single image or
                           (T, H, W, C) for a video sequence.
            pred_seq     : numpy array with the same layout as true_seq.
        """
        self.reset_per_sequence()

        # Initialise an empty list for every metric in this sequence
        self.seq_idx_curr = seq
        self.metric_dict[self.seq_idx_curr] = OrderedDict({
            metric: [] for metric in self.metric_opt.keys()})

        # Determine video vs single-image and normalize shape to (T, H, W, C)
        if true_seq is not None:
            if len(true_seq.shape) == 4:    # (B, H, W, C) — single image
                self.is_video = False
            else:                               # (B, T, H, W, C) — video
                true_seq = true_seq.squeeze(0)  # Processing one video at a time
                pred_seq = pred_seq.squeeze(0)
                self.is_video = True
        else:
            # Fall back to reading frames from disk in sorted order
            true_img_lst = base_utils.retrieve_files(true_seq_dir, 'png')
            pred_img_lst = base_utils.retrieve_files(pred_seq_dir, 'png')
            self.is_video = len(true_img_lst) > 1

        num_frames = (len(true_seq) if true_seq is not None
                      else len(true_img_lst))

        for i in range(num_frames):
            # Load current frame — from array or from disk
            if true_seq is not None:
                self.true_img_cur = true_seq[i]   # HWC | RGB | uint8
            else:
                self.true_img_cur = cv2.imread(
                    true_img_lst[i], cv2.IMREAD_UNCHANGED)[..., ::-1]  # BGR->RGB

            if pred_seq is not None:
                self.pred_img_cur = pred_seq[i]   # HWC | RGB | uint8
            else:
                self.pred_img_cur = cv2.imread(
                    pred_img_lst[i])[..., ::-1]

            # Per-frame metrics
            self.compute_frame_metrics()

            # Keep previous frames for temporal metrics on the next iteration
            self.true_img_pre = self.true_img_cur
            self.pred_img_pre = self.pred_img_cur

        # sigma_ids is a sequence-level metric — compute it after all
        # frames have been processed (needs the full IDS list).
        if 'Sigma_IDS' in self.metric_opt:
            ids_values = self.metric_dict[self.seq_idx_curr].get('IDS', [])
            sigma = float(np.std(ids_values)) if len(ids_values) >= 2 else 0.0
            self.metric_dict[self.seq_idx_curr]['Sigma_IDS'] = [sigma]

    # ── Per-frame dispatch ────────────────────────────────────────────────────

    def compute_frame_metrics(self):
        metric_dict = self.metric_dict[self.seq_idx_curr]

        for metric_type, opt in self.metric_opt.items():

            if metric_type == 'PSNR':
                PSNR = self.compute_PSNR()
                if not np.isinf(PSNR):          # skip perfect-match frames
                    metric_dict['PSNR'].append(PSNR)

            elif metric_type == 'SSIM':
                metric_dict['SSIM'].append(self.compute_SSIM())

            elif metric_type == 'LPIPS':
                LPIPS = self.compute_LPIPS()[0, 0, 0, 0].cpu().numpy()
                metric_dict['LPIPS'].append(LPIPS)

            elif metric_type == 'IDS':
                metric_dict['IDS'].append(self.compute_IDS())

            elif metric_type == 'IDSRef':
                pass    # to be implemented

            # sigma_ids is accumulated at sequence level (after the
            # frame loop), so there is nothing to do per-frame here.
            elif metric_type == 'sigma_ids':
                pass

            elif metric_type == 'NIQE':
                metric_dict['NIQE'].append(self.compute_NIQE())

            elif metric_type == 'MUSIQ':
                metric_dict['MUSIQ'].append(self.compute_MUSIQ())

            elif metric_type == 'CLIPIQA':
                metric_dict['CLIPIQA'].append(self.compute_CLIPIQA())

            elif metric_type == 'tOF':
                # tOF requires two consecutive frames — skip the first one
                if self.is_video and self.pred_img_pre is not None:
                    metric_dict['tOF'].append(self.compute_tOF())

            elif metric_type == 'EWarp':
                # EWarp requires two consecutive frames — skip the first one
                if self.is_video and self.pred_img_pre is not None:
                    metric_dict['EWarp'].append(self.compute_EWarp())

    # ── Individual metric implementations ────────────────────────────────────

    def compute_PSNR(self):
        if self.psnr_colorspace == 'rgb':
            true_img = self.true_img_cur
            pred_img = self.pred_img_cur
        else:
            # convert to ycbcr, and keep the y channel
            true_img = data_utils.rgb_to_ycbcr(self.true_img_cur)[..., 0]
            pred_img = data_utils.rgb_to_ycbcr(self.pred_img_cur)[..., 0]

        diff = true_img.astype(np.float64) - pred_img.astype(np.float64)
        RMSE = np.sqrt(np.mean(np.power(diff, 2)))

        if RMSE == 0:
            return np.inf

        PSNR = 20 * np.log10((2 ** self.gt_bit_depth - 1) / RMSE)
        return PSNR

    def compute_SSIM(self):
        # Convert to YCbCr and keep only the luma channel
        true_img = data_utils.rgb_to_ycbcr(self.true_img_cur)[..., 0]
        pred_img = data_utils.rgb_to_ycbcr(self.pred_img_cur)[..., 0]

        ssim_val = ssim(
            true_img, pred_img,
            data_range=(2 ** self.gt_bit_depth - 1),
            multichannel=False,
        )
        return ssim_val

    def compute_LPIPS(self):
        if isinstance(self.true_img_cur, np.ndarray):
            true_img = np.ascontiguousarray(self.true_img_cur)
            pred_img = np.ascontiguousarray(self.pred_img_cur)

            # HWC uint8 -> 1CHW float in [0, 1]
            true_img = torch.FloatTensor(true_img).unsqueeze(0).permute(0, 3, 1, 2) / 255.
            pred_img = torch.FloatTensor(pred_img).unsqueeze(0).permute(0, 3, 1, 2) / 255.

            # Normalise to [-1, 1] as expected by LPIPS
            normalize(true_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(pred_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        else:
            true_img = self.true_img_cur.unsqueeze(0)
            pred_img = self.pred_img_cur.unsqueeze(0)

        with torch.no_grad():
            LPIPS = self.dm.forward(true_img, pred_img)

        return LPIPS

    def compute_IDS(self):
        if isinstance(self.true_img_cur, np.ndarray):
            # HWC uint8 -> 1CHW float, then normalise to [-1, 1]
            true_img = torch.FloatTensor(self.true_img_cur).unsqueeze(0).permute(0, 3, 1, 2) / 255.
            pred_img = torch.FloatTensor(self.pred_img_cur).unsqueeze(0).permute(0, 3, 1, 2) / 255.
            true_img = normalize(true_img, [0.5] * 3, [0.5] * 3)
            pred_img = normalize(pred_img, [0.5] * 3, [0.5] * 3)
        else:
            true_img = self.true_img_cur.unsqueeze(0)
            pred_img = self.pred_img_cur.unsqueeze(0)

        with torch.no_grad():
            true_emb = self.net_ID(true_img.to(self.device))
            pred_emb = self.net_ID(pred_img.to(self.device))

            # Cosine similarity between the two identity embeddings
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            ids = cos(true_emb, pred_emb).cpu().numpy()[0]

        return ids

    @staticmethod
    def _to_uint8_hwc(img) -> np.ndarray:
        """Ensure a frame is a contiguous (H, W, C) uint8 numpy array.

        Accepts:
            - numpy (H, W, C) uint8   — returned as-is (contiguous copy)
            - numpy (H, W, C) float   — assumed [0, 1], scaled to [0, 255]
            - torch (C, H, W) float   — assumed [-1, 1], rescaled then permuted
            - torch (H, W, C) float   — assumed [-1, 1], rescaled
        """
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().float()
            # Detect layout: if first dim is 3 (C=3) and last dim is not 3,
            # assume CHW; otherwise assume HWC.
            if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
                img = img.permute(1, 2, 0)          # CHW -> HWC
            # Rescale [-1, 1] -> [0, 255]
            img = ((img + 1.0) * 127.5).clamp(0, 255)
            img = img.numpy().astype(np.uint8)
        elif img.dtype != np.uint8:
            # Float numpy assumed [0, 1]
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    def compute_tOF(self):
        # Ensure all four frames are (H, W, C) uint8 numpy arrays
        true_cur = self._to_uint8_hwc(self.true_img_cur)
        pred_cur = self._to_uint8_hwc(self.pred_img_cur)
        true_pre = self._to_uint8_hwc(self.true_img_pre)
        pred_pre = self._to_uint8_hwc(self.pred_img_pre)

        # Convert all four frames to grayscale for optical-flow estimation
        true_cur = cv2.cvtColor(true_cur, cv2.COLOR_RGB2GRAY)
        pred_cur = cv2.cvtColor(pred_cur, cv2.COLOR_RGB2GRAY)
        true_pre = cv2.cvtColor(true_pre, cv2.COLOR_RGB2GRAY)
        pred_pre = cv2.cvtColor(pred_pre, cv2.COLOR_RGB2GRAY)

        # Dense optical flow (Farneback) for GT and prediction separately
        true_OF = cv2.calcOpticalFlowFarneback(
            true_pre, true_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        pred_OF = cv2.calcOpticalFlowFarneback(
            pred_pre, pred_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # End-point error (EPE) between GT flow and predicted flow
        diff_OF = true_OF - pred_OF
        tOF = np.mean(np.sqrt(np.sum(diff_OF ** 2, axis=-1)))
        return tOF

    def compute_EWarp(self):
        """Estimated Warp error — temporal consistency metric.

        Warps the previous predicted frame to the current time step using
        dense optical flow estimated on the GT sequence, then measures the
        MAE against the actual current predicted frame.  Any residual error
        is caused by temporal inconsistency in the model output, not by
        real scene motion.

        Lower EWarp  ->  more temporally stable predictions.

        Reference: Lai et al., "Learning Blind Video Temporal Consistency"
                   (ECCV 2018).
        """
        # Ensure all frames are (H, W, C) uint8 numpy arrays
        true_pre = self._to_uint8_hwc(self.true_img_pre)
        true_cur = self._to_uint8_hwc(self.true_img_cur)
        pred_pre = self._to_uint8_hwc(self.pred_img_pre)
        pred_cur = self._to_uint8_hwc(self.pred_img_cur)

        # Estimate dense GT flow (prev -> cur) on grayscale frames
        true_gray_pre = cv2.cvtColor(true_pre, cv2.COLOR_RGB2GRAY)
        true_gray_cur = cv2.cvtColor(true_cur, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            true_gray_pre, true_gray_cur,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )   # (H, W, 2)

        h, w = flow.shape[:2]

        # Build absolute destination coordinate maps
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)

        # Warp the previous predicted frame with the GT flow
        warped_bgr = cv2.remap(
            pred_pre[..., ::-1],    # RGB -> BGR for cv2.remap
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warped_rgb = warped_bgr[..., ::-1]  # BGR -> RGB

        # MAE between warped previous frame and current predicted frame
        diff = warped_rgb.astype(np.float64) - pred_cur.astype(np.float64)
        return float(np.mean(np.abs(diff)))

    def compute_NIQE(self):
        # Convert to 1×3×H×W tensor in [0, 1] as expected by pyiqa
        if isinstance(self.pred_img_cur, np.ndarray):
            img = torch.FloatTensor(self.pred_img_cur).permute(2, 0, 1).unsqueeze(0) / 255.
        else:
            img = self.pred_img_cur.unsqueeze(0)
            img = (img + 1) / 2         # [-1, 1] -> [0, 1]
            img = img.clamp(0, 1)

        img = img.to(self.device)

        with torch.no_grad():
            score = self.niqe_model(img)

        return score.item()

    def compute_MUSIQ(self):
        # Convert to 1×3×H×W tensor in [0, 1] as expected by pyiqa
        if isinstance(self.pred_img_cur, np.ndarray):
            img = (torch.FloatTensor(self.pred_img_cur)
                   .permute(2, 0, 1).unsqueeze(0) / 255.)
        else:
            img = self.pred_img_cur.unsqueeze(0)
            img = (img + 1) / 2
            img = img.clamp(0, 1)

        img = img.to(self.device)

        with torch.no_grad():
            score = self.musiq_model(img)

        return score.item()

    def compute_CLIPIQA(self):
        # Convert to 1×3×H×W tensor in [0, 1] as expected by pyiqa
        if isinstance(self.pred_img_cur, np.ndarray):
            img = (torch.FloatTensor(self.pred_img_cur)
                   .permute(2, 0, 1).unsqueeze(0) / 255.)
        else:
            img = self.pred_img_cur.unsqueeze(0)
            img = (img + 1) / 2
            img = img.clamp(0, 1)

        img = img.to(self.device)

        with torch.no_grad():
            score = self.clipiqa_model(img)

        return score.item()