import os
import os.path as osp
import scipy
import cv2
import math
import numpy as np
import torch
import random
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from PIL import Image
from basicsr.data import degradations
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp
from torchvision.transforms.functional import rgb_to_grayscale


import io as _io

try:
    import av as _av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False


kernel_range = [2 * v + 1 for v in range(3, 11)]

# Lazily-initialised CUDA helpers — instantiated on first use so that merely
# importing this module does not force CUDA initialisation.
_jpeger = None          # DiffJPEG: simulate JPEG compression artifacts
_usm_sharpener = None    # USMSharp: sharpening instance


def _get_jpeger():
    global _jpeger
    if _jpeger is None:
        _jpeger = DiffJPEG(differentiable=False).cuda()
    return _jpeger


def _get_usm_sharpener():
    global _usm_sharpener
    if _usm_sharpener is None:
        _usm_sharpener = USMSharp().cuda()
    return _usm_sharpener

def rgb_to_ycbcr(img):
    """ Coefficients are taken from the  official codes of DUF-VSR
        This conversion is also the same as that in BasicSR

        Parameters:
            :param  img: rgb image in type np.uint8
            :return: ycbcr image in type np.uint8
    """

    T = np.array([
        [0.256788235294118, -0.148223529411765,  0.439215686274510],
        [0.504129411764706, -0.290992156862745, -0.367788235294118],
        [0.097905882352941,  0.439215686274510, -0.071427450980392],
    ], dtype=np.float64)

    O = np.array([16, 128, 128], dtype=np.float64)

    img = img.astype(np.float64)
    res = np.matmul(img, T) + O
    res = res.clip(0, 255).round().astype(np.uint8)

    return res


def save_sequence(seq_dir, seq_data, frm_idx_lst=None, to_bgr=False, dtype=np.uint8):
    """ Save each frame of a sequence to .png image in seq_dir

        Parameters:
            :param seq_dir: dir to save results
            :param seq_data: sequence with shape thwc|uint8
            :param frm_idx_lst: specify filename for each frame to be saved
            :param to_bgr: whether to flip color channels
    """

    if to_bgr:
        seq_data = seq_data[..., ::-1]  # rgb2bgr

    # use default frm_idx_lst is not specified
    tot_frm = len(seq_data)
    if frm_idx_lst is None:
        frm_idx_lst = ['{:04d}.png'.format(i) for i in range(tot_frm)]

    # save for each frame
    os.makedirs(seq_dir, exist_ok=True)
    for i in range(tot_frm):
        cv2.imwrite(osp.join(seq_dir, frm_idx_lst[i]), seq_data[i].astype(dtype))

#--- Image Processing Functions ---#
def post_process(img, bit_depth=8, rgb2bgr=False, min=-1, max=1):
    # Collapse batch dim for 5D video tensors
    if img.ndim == 5:
        assert img.shape[0] == 1, \
            f'post_process: batch size must be 1 for 5D input, got {img.shape[0]}'
        img = img[0]    # (T, C, H, W)
        
    # Clip and normalize to [0, 1]
    img = torch.clamp_(img, min, max)
    img = (img - min) / (max - min + 1e-7)
    
    # Convert to int
    img = img * (2 ** bit_depth - 1)
    dtype = np.uint8 if bit_depth == 8 else np.uint16

    # Transform to numpy HxWxC
    if len(img.shape) == 4:
        img = img.permute(0, 2, 3, 1).cpu().numpy().astype(dtype)
    else: 
        img = img.permute(1, 2, 0).cpu().numpy().astype(dtype)

    # Convert to BGR if needed
    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img



@torch.no_grad()
def sample_degraded_image(opt, gt, lr=None, ref=None, kernel=None, has_lr=False):
    """ prepare gt, lr data for training

        for BD/Synth degradation, generate lr data and remove border of gt data
        for BI degradation, return data directly

    """
    device = torch.device(opt['device'])
    degradation_type = opt['dataset']['degradation']['type']

    # Move on device
    gt = gt.to(device)
    if has_lr:        
        # Check if LR is provided, otherwise apply the degradation pipeline
        if lr is None:
            raise Exception('No LR data provided by dataset.')
        else:
            lr = lr.to(device) 
    
    elif degradation_type.lower() == 'synth':
        # TODO: fix this line back, so degradation is applied
        if lr is None:
            lr, gt = apply_degradation_pipeline(opt['dataset']['train'], gt, scale=opt['scale'])
        else:
            lr = lr.to(device)
        normalize(lr, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        normalize(gt, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        if len(ref) > 0:
            # Normalize does not support 5-dim layout
            mean = ref.new_tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
            std  = ref.new_tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
            ref.sub_(mean).div_(std)

    elif degradation_type.lower() == 'bd':
        # setup params
        scale = opt['scale']
        sigma = opt['dataset']['degradation'].get('sigma', 1.5)
        kernel = create_bd_kernel(sigma)
        border_size = int(sigma * 3.0)

        gt_with_border = gt
        n, t, c, gt_h, gt_w = gt_with_border.size()
        lr_h = (gt_h - 2 * border_size) // scale
        lr_w = (gt_w - 2 * border_size) // scale

        # generate lr data
        gt_with_border = gt_with_border.view(n * t, c, gt_h, gt_w)
        lr = F.conv2d(
            gt_with_border, kernel, stride=scale, bias=None, padding=0)
        lr = lr.view(n, t, c, lr_h, lr_w)

        # remove gt border
        gt = gt_with_border[
            ...,
            border_size: border_size + scale * lr_h,
            border_size: border_size + scale * lr_w
        ]
        gt = gt.view(n, t, c, scale * lr_h, scale * lr_w)

    else:
        raise ValueError('Unrecognized degradation type: {}'.format(
            degradation_type))

    return {
        'gt': gt, 
        'lr': lr,
        'ref': ref
    }

# --- BD kernel --- #
def create_bd_kernel(sigma):
    ksize = 1 + 2 * int(sigma * 3.0)

    # gkern1d = signal.gaussian(ksize, std=sigma).reshape(ksize, 1)
    x = np.linspace(-ksize // 2, ksize // 2, ksize)
    gkern1d = scipy.stats.norm.pdf(x, 0, sigma).reshape(ksize, 1)  # Corrected usage
    gkern1d /= gkern1d.sum()  # Normalize
    gkern2d = np.outer(gkern1d, gkern1d)
    gaussian_kernel = gkern2d / gkern2d.sum()
    zero_kernel = np.zeros_like(gaussian_kernel)

    kernel = np.float32([
        [gaussian_kernel, zero_kernel, zero_kernel],
        [zero_kernel, gaussian_kernel, zero_kernel],
        [zero_kernel, zero_kernel, gaussian_kernel]])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = torch.from_numpy(kernel).to(device)

    return kernel

def get_random_resampling_mode():
    return random.choice(['area', 'bilinear', 'bicubic'])

def get_random_resample_type(resize_prob):
    # resize_prob = [p_up, p_down, p_keep]
    return np.random.choice(['up', 'down', 'keep'], p=resize_prob)

def get_random_resample_scale(resize_type, resize_range):
    if resize_type == 'up':
        return np.random.uniform(1.0, resize_range[1])
    elif resize_type == 'down':
        return np.random.uniform(resize_range[0], 1.0)
    else:  # keep
        return 1.0

@torch.no_grad()
def apply_degradation_pipeline(data_opt, gt, scale=1, debug=False):   
    # Debug
    if debug:
        print(f"GT shape: {gt.shape}.")
    
    # Apply first degradation
    first_degradation_opt = data_opt.get('first_degradation', {})
    lr = apply_degradation_kernel(gt, first_degradation_opt)
    
    # Apply second degradation
    second_degradation_opt = data_opt.get('second_degradation', None)
    if second_degradation_opt is not None:
        lr = apply_degradation_kernel(lr, second_degradation_opt)
    
    # Resize to desired size
    resize_mode = get_random_resampling_mode()
    gt_h, gt_w = gt.shape[-2:]
    lr = F.interpolate(lr, size=(gt_h // scale, gt_w // scale), 
                       mode=resize_mode)

    # Apply final sinc
    if torch.rand(1) < data_opt.get('final_sinc_prob', 0.8):
        if debug:
            print("Applying final sinc kernel...")
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel).unsqueeze(dim=0).cuda()
        lr = filter2D(lr, sinc_kernel)
        
    # Clamp and round
    lr = torch.clamp((lr * 255.0).round(), 0, 255) / 255.

    # Sharpen gt
    if torch.rand(1) < data_opt.get('sharpen_gt_prob', 0):
        if debug:
            print("Applying sharpening to GT...")
        gt = _get_usm_sharpener()(gt)
            
    # Return sequences
    return lr, gt
                
def apply_degradation_kernel(img,
                             degradation_opt,
                             debug=False):   
    # LR shape
    if len(img.shape) == 5:
        # Video
        B, T, C, H, W = img.shape
    elif len(img.shape) == 4:
        # Image
        B, C, H, W = img.shape
        T = None
    else:
        raise Exception("Input shape must have at least 4 dims.")

    # Apply blur
    if torch.rand(1) <= degradation_opt['blur_prob']:
        # Define the degradation kernel and apply it
        if debug:
            print("Applying blur...")
        degradation_kernel = define_blur_kernel(degradation_opt)
        img = filter2D(img, degradation_kernel)
    
    # Rescale
    resampling_mode = get_random_resampling_mode()
    resampling_type = get_random_resample_type(degradation_opt['resize_prob'])
    resampling_scale = get_random_resample_scale(resize_type=resampling_type,
                                                 resize_range=degradation_opt['resize_range'])
    img = F.interpolate(img, scale_factor=resampling_scale, mode=resampling_mode)
    
    # Apply Gaussian/Posisson noise 
    if torch.rand(1) < degradation_opt['gaussian_noise_prob']:
        if debug:
            print("Applying gaussian...")
        img = random_add_gaussian_noise_pt(img, 
                                            sigma_range=degradation_opt['noise_range'],
                                            clip=True, 
                                            rounds=False,
                                            gray_prob=degradation_opt['gray_noise_prob'])
    elif torch.rand(1) < degradation_opt['poisson_noise_prob']:
        if debug:
            print("Applying poisson...")
        img = random_add_poisson_noise_pt(img,
                                          scale_range=degradation_opt['poisson_scale_range'],
                                          gray_prob=degradation_opt['gray_noise_prob'],
                                          clip=True,
                                          rounds=False)
    
    # JPEG compression
    if torch.rand(1) <= degradation_opt['jpeg_prob']:
        jpeg_qp = img.new_zeros(img.size(0)).uniform_(*degradation_opt['jpeg_range'])
        img = torch.clamp(img, 0, 1)
        img = _get_jpeger()(img, quality=jpeg_qp)
        if debug:
            print(f"Applying jpeg: {jpeg_qp}.")

    # Return output
    return img

# ----- Degradation Functions ----- #
def define_blur_kernel(degradation_opt):
    kernel_size = degradation_opt.get('blur_kernel_size', None)
    if kernel_size is None:
        kernel_size = random.choice(kernel_range)
        pad_size = (kernel_range[-1] - kernel_size) // 2
    else:
        pad_size = 0
    if torch.rand(1) < degradation_opt['sinc_prob']:
        # this sinc filter setting is for kernels ranging from [7, 21]
        low = np.pi / 3 if kernel_size < 13 else np.pi / 5
        omega_c = torch.rand(1).mul(np.pi - low).add(low).item()
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            degradation_opt['kernel_list'],
            degradation_opt['kernel_prob'],
            kernel_size,
            degradation_opt['blur_sigma'],
            degradation_opt['blur_sigma'], 
            [-math.pi, math.pi],
            degradation_opt['betag_range'],
            degradation_opt['betap_range'],
            noise_range=None)
        
    # Pad kernel
    if pad_size != 0:
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    
    # Return kernel
    kernel = torch.FloatTensor(kernel).unsqueeze(dim=0).cuda()
    return kernel

def pre_process_data(data_dict):
    # Fetch data
    lr = data_dict['lr']
    gt = data_dict.get('gt')
    ref = data_dict.get('ref')
    return_dict = {}
    
    # Video datasets normalize in __getitem__ already — skip double normalization
    is_video = (lr.dim() == 5)  # (B, T, C, H, W) = video, (B, C, H, W) = image

    if not is_video:
        normalize(lr, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        if gt is not None:
            normalize(gt, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        if ref is not None:
            mean = ref.new_tensor([0.5,0.5,0.5]).view(1,1,3,1,1)
            std  = ref.new_tensor([0.5,0.5,0.5]).view(1,1,3,1,1)
            ref.sub_(mean).div_(std)
            return_dict['ref'] = ref

    return_dict['lr'] = lr
    if gt is not None:
        return_dict['gt'] = gt

    return return_dict


# =============================================================================
# Video degradation pipeline
# =============================================================================
# Mirrors the KEEP / VFHQDataset degradation order:
#   blur (consistent per clip) -> resize -> noise (per frame) -> CRF (consistent per clip)
#
# Config compatibility: identical keys to the existing image pipeline, with
# jpeg_prob / jpeg_range replaced by crf_prob / crf_range / crf_codecs / crf_codec_probs.
 
# ----------------- Filter 2D Video Sequence --------------------- #
def filter2D_video(video, kernel):
    """Apply 2D spatial filtering per video in a batch of sequences.
 
    Args:
        video (Tensor): (B, T, C, H, W)
        kernel (Tensor): (B, K, K) - one kernel per video
    Returns:
        Tensor: (B, T, C, H, W)
    """
    B, T, C, H, W = video.shape
    K = kernel.size(-1)
 
    if K % 2 == 0:
        raise ValueError("Kernel size must be odd")
 
    # Reshape to 4D for padding
    video = video.view(B * T, C, H, W)
 
    # Pad spatial dims using reflect mode (now 4D input)
    video = F.pad(video, (K // 2, K // 2, K // 2, K // 2), mode='reflect')
 
    if kernel.size(0) == 1:
        # One shared kernel for the whole batch
        kernel = kernel.view(1, 1, K, K)
        video = video.view(B * T * C, 1, H + K - 1, W + K - 1)
        out = F.conv2d(video, kernel, padding=0)
        return out.view(B, T, C, H, W)
    else:
        # Per-video kernel: repeat each kernel T times, then replicate over channels
        kernel = kernel.unsqueeze(1).repeat(1, T, 1, 1).view(B * T, 1, K, K)
        kernel = kernel.repeat_interleave(C, dim=0)                 # (B*T*C, 1, K, K)
        video = video.view(B * T * C, 1, H + K - 1, W + K - 1)
        out = F.conv2d(video, kernel, groups=B * T * C)
        return out.view(B, T, C, H, W)
 
 
# --------------------- Noise functions (image + video) --------------------- #
 
def generate_gaussian_noise_pt(img, sigma=10, gray_noise=False):
    """Generate Gaussian noise for 4D (B,C,H,W) or 5D (B,T,C,H,W) tensors.
 
    Args:
        img (Tensor): Input tensor, values in [0, 1].
        sigma (float | Tensor): Std-dev in [0, 255]. Shape (B,) for per-batch.
        gray_noise (bool | Tensor): 1 = grayscale noise. Shape (B,) for per-batch.
    Returns:
        Tensor: Noise tensor, same shape as input.
    """
    noise = torch.randn_like(img)
 
    # Normalise sigma to [0, 1] and reshape to broadcast over all spatial dims
    if isinstance(sigma, torch.Tensor):
        sigma = sigma / 255.0
        if sigma.dim() == 1:
            # (B,) -> (B, 1, ..., 1)  — works for both 4D and 5D input
            extra_dims = img.dim() - 1
            sigma = sigma.view(-1, *([1] * extra_dims))
    else:
        sigma = torch.tensor(sigma / 255.0, dtype=img.dtype, device=img.device)
 
    if isinstance(gray_noise, torch.Tensor):
        extra_dims = img.dim() - 1
        gray_noise = gray_noise.view(-1, *([1] * extra_dims)).float()
        # Channel dim is always dim=1 in PyTorch channels-first tensors (BCHW / BTCHW)
        noise_gray = noise.mean(dim=1, keepdim=True).expand_as(noise)
        noise = gray_noise * noise_gray + (1 - gray_noise) * noise
    elif gray_noise:
        noise = noise.mean(dim=1, keepdim=True).expand_as(noise)
 
    return noise * sigma
 
 
def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    """Sample sigma and gray flag, then call generate_gaussian_noise_pt.
 
    For 4D input: one sigma and gray flag for the whole batch.
    For 5D input: one sigma and gray flag per sequence (consistent per clip).
    """
    if img.dim() == 4:
        sigma      = torch.empty(1, dtype=img.dtype, device=img.device).uniform_(*sigma_range)
        gray_noise = (torch.rand(1, dtype=img.dtype, device=img.device) < gray_prob).float()
    elif img.dim() == 5:
        B          = img.size(0)
        sigma      = torch.empty(B, dtype=img.dtype, device=img.device).uniform_(*sigma_range)
        gray_noise = (torch.rand(B, dtype=img.dtype, device=img.device) < gray_prob).float()
    else:
        raise ValueError(f"Unsupported tensor dimensionality: {img.dim()}D")
 
    return generate_gaussian_noise_pt(img, sigma, gray_noise)
 
 
def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    """Add randomly-sampled Gaussian noise. Supports 4D and 5D tensors."""
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out   = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out
 
 
def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    """Generate Poisson noise for 4D (B,C,H,W) or 5D (B,T,C,H,W) tensors.
 
    Args:
        img (Tensor): Input, range [0, 1], float32.
        scale (float | Tensor): Noise scale. Shape (B,) for per-sequence.
        gray_noise (float | Tensor): 0-1. Shape (B,) for per-sequence.
    Returns:
        Tensor: Noise tensor, same shape as input.
    """
    is_5d = img.ndim == 5
    if is_5d:
        B, T, C, H, W = img.shape
        img_4d = img.view(B * T, C, H, W)
    else:
        B, C, H, W = img.shape
        T    = 1
        img_4d = img
    N = B * T
 
    # Expand scale and gray_noise from per-sequence (B,) to per-frame (B*T,)
    def _expand(x, default):
        if isinstance(x, torch.Tensor):
            if x.numel() == B:
                return x.view(B, 1).expand(B, T).reshape(N, 1, 1, 1)
            elif x.numel() == N:
                return x.view(N, 1, 1, 1)
            else:
                raise ValueError(f"Unexpected tensor size {x.shape} for B={B}, T={T}")
        return torch.tensor(default if not isinstance(x, (int, float)) else x,
                            device=img.device, dtype=img.dtype).expand(N, 1, 1, 1)
 
    scale_4d      = _expand(scale, 1.0)
    gray_noise_4d = _expand(gray_noise, 0)
    cal_gray      = (gray_noise_4d > 0).any()
 
    if cal_gray:
        img_gray = rgb_to_grayscale(img_4d, num_output_channels=1)
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        vals_list  = [len(torch.unique(img_gray[i])) for i in range(N)]
        vals       = img_gray.new_tensor([2 ** np.ceil(np.log2(v)) for v in vals_list]).view(N, 1, 1, 1)
        out_gray   = torch.poisson(img_gray * vals) / vals
        noise_gray = (out_gray - img_gray).expand(N, C, H, W)
 
    img_4d_clamp = torch.clamp((img_4d * 255.0).round(), 0, 255) / 255.
    vals_list = [len(torch.unique(img_4d_clamp[i])) for i in range(N)]
    vals      = img_4d_clamp.new_tensor([2 ** np.ceil(np.log2(v)) for v in vals_list]).view(N, 1, 1, 1)
    out       = torch.poisson(img_4d_clamp * vals) / vals
    noise     = out - img_4d_clamp
 
    if cal_gray:
        noise = noise * (1 - gray_noise_4d) + noise_gray * gray_noise_4d
 
    noise = noise * scale_4d
 
    if is_5d:
        noise = noise.view(B, T, C, H, W)
    return noise
 
 
def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0.0):
    """Sample scale and gray flag, then call generate_poisson_noise_pt.
 
    Scale and gray flag are sampled per sequence (B,) for temporal consistency.
    """
    is_5d = img.ndim == 5
    B     = img.size(0)
 
    scale      = torch.rand(B, dtype=img.dtype, device=img.device) \
                 * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = (torch.rand(B, dtype=img.dtype, device=img.device) < gray_prob).float()
    return generate_poisson_noise_pt(img, scale, gray_noise)
 
 
def random_add_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0.0, clip=True, rounds=False):
    """Add randomly-sampled Poisson noise. Supports 4D and 5D tensors."""
    noise = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out   = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out
 
 
# --------------------- CRF compression ------------------------------------ #
 
def _apply_crf_to_sequence(clip_tchw: torch.Tensor, crf: int, codec: str) -> torch.Tensor:
    """Apply CRF video compression to a single clip using PyAV (in-memory).
 
    Args:
        clip_tchw: Float tensor (T, C, H, W) in [0, 1], RGB, on any device.
        crf:       Constant Rate Factor. Higher = more compression.
        codec:     FFmpeg video codec string, e.g. 'libx264' or 'libx265'.
    Returns:
        Float tensor (T, C, H, W) in [0, 1], RGB, same device as input.
    """
    if not _AV_AVAILABLE:
        raise RuntimeError(
            "PyAV is required for CRF compression. Install with: pip install av"
        )
 
    T, C, H, W = clip_tchw.shape
    device      = clip_tchw.device
 
    # Float [0,1] RGB -> uint8 numpy (T, H, W, 3)
    frames_np = (
        clip_tchw.permute(0, 2, 3, 1).cpu().numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
 
    buf = _io.BytesIO()
 
    # Encode
    with _av.open(buf, 'w', 'mp4') as container:
        stream         = container.add_stream(codec, rate=1)
        stream.height  = H
        stream.width   = W
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': str(crf)}
 
        for frm_np in frames_np:
            av_frame           = _av.VideoFrame.from_ndarray(frm_np, format='rgb24')
            av_frame.pict_type = 0
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():  # flush encoder
            container.mux(packet)
 
    # Decode
    buf.seek(0)
    decoded = []
    with _av.open(buf, 'r', 'mp4') as container:
        for frame in container.decode(video=0):
            decoded.append(frame.to_rgb().to_ndarray())  # (H, W, 3) uint8
 
    # Guard against codec length drift at clip boundaries
    while len(decoded) < T:
        decoded.append(decoded[-1])
    decoded = decoded[:T]
 
    out_np = np.stack(decoded).astype(np.float32) / 255.0  # (T, H, W, 3)
    out    = torch.from_numpy(np.ascontiguousarray(out_np))
    out    = out.permute(0, 3, 1, 2).to(device)            # (T, C, H, W)
    return out
 
 
# --------------------- Video degradation helpers -------------------------- #
 
def _add_noise_video(img, degradation_opt, debug=False):
    """Apply one round of Gaussian or Poisson noise to (B, T, C, H, W).
 
    Uses the project's own random_add_gaussian_noise_pt /
    random_add_poisson_noise_pt which are 5D-aware (defined above).
    """
    if torch.rand(1).item() < degradation_opt['gaussian_noise_prob']:
        if debug:
            print("  Applying gaussian noise...")
        img = random_add_gaussian_noise_pt(
            img,
            sigma_range=degradation_opt['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=degradation_opt['gray_noise_prob'],
        )
    elif torch.rand(1).item() < degradation_opt['poisson_noise_prob']:
        if debug:
            print("  Applying poisson noise...")
        img = random_add_poisson_noise_pt(
            img,
            scale_range=degradation_opt['poisson_scale_range'],
            gray_prob=degradation_opt['gray_noise_prob'],
            clip=True,
            rounds=False,
        )
    return img
 
 
def apply_degradation_kernel_video(img: torch.Tensor,
                                   degradation_opt: dict,
                                   debug: bool = False) -> torch.Tensor:
    """Apply one degradation stage to a video clip (B, T, C, H, W).
 
    Pipeline order:
        1. Blur   — kernel sampled ONCE per clip (same for all frames).
        2. Resize — scale sampled once, consistent across all frames.
                    H/W rounded to nearest even number for yuv420p CRF.
        3. Noise  — per-frame (temporally independent sensor noise).
        4. CRF    — codec + CRF sampled once per clip, encoded via PyAV.
 
    Args:
        img:             Float tensor (B, T, C, H, W) in [0, 1] on CUDA.
        degradation_opt: Degradation stage config dict.
        debug:           Print applied steps to stdout.
    Returns:
        Float tensor (B, T, C, H, W) in [0, 1].
    """
    if img.ndim != 5:
        raise ValueError(
            f"apply_degradation_kernel_video expects 5-D (B,T,C,H,W), "
            f"got {img.ndim}-D."
        )
 
    B, T, C, H, W = img.shape
 
    # 1. Blur — one kernel per batch, shared across all T frames
    if torch.rand(1).item() <= degradation_opt.get('blur_prob', 0.8):
        if debug:
            print("  Applying blur...")
        blur_kernel = define_blur_kernel(degradation_opt)   # (1, K, K)
        img = filter2D_video(img, blur_kernel)
 
    # 2. Resize — consistent scale across all frames
    resampling_mode  = get_random_resampling_mode()
    resampling_type  = get_random_resample_type(degradation_opt['resize_prob'])
    resampling_scale = get_random_resample_scale(
        resize_type=resampling_type,
        resize_range=degradation_opt['resize_range'],
    )
    if resampling_scale != 1.0:
        target_h = max(2, int(H * resampling_scale) // 2 * 2)  # force even for yuv420p
        target_w = max(2, int(W * resampling_scale) // 2 * 2)
        flat = img.view(B * T, C, H, W)
        flat = F.interpolate(flat, size=(target_h, target_w), mode=resampling_mode)
        img  = flat.view(B, T, C, target_h, target_w)
        if debug:
            print(f"  Resize ×{resampling_scale:.3f} -> ({target_h}, {target_w})")
 
    # 3. Noise — per-frame, temporally independent
    img = _add_noise_video(img, degradation_opt, debug=debug)
 
    # 4. CRF compression — consistent per clip
    crf_prob = degradation_opt.get('crf_prob', 0.0)
    if torch.rand(1).item() <= crf_prob:
        crf_range       = degradation_opt.get('crf_range',  [25, 45])
        crf_codecs      = degradation_opt.get('crf_codecs', ['libx264'])
        crf_codec_probs = degradation_opt.get('crf_codec_probs', None)
 
        # Replace decoder alias with the correct encoder name
        crf_codecs = ['libx264' if c == 'h264' else c for c in crf_codecs]
 
        codec = random.choices(crf_codecs, weights=crf_codec_probs, k=1)[0]
        img   = torch.clamp(img, 0, 1)
 
        compressed = []
        for b in range(B):
            crf_val = random.randint(crf_range[0], crf_range[1])
            if debug:
                print(f"  CRF {crf_val} ({codec}) on clip {b}...")
            compressed.append(
                _apply_crf_to_sequence(img[b], crf=crf_val, codec=codec)
            )
        img = torch.stack(compressed)
 
    # 5. Upsample back to original shape
    if img.shape[-2:] != (H, W):
        flat = img.view(B * T, C, *img.shape[-2:])
        flat = F.interpolate(flat, size=(H, W), mode='bilinear', align_corners=False)
        img  = flat.view(B, T, C, H, W)

    return img
 
 
    return lr, gt