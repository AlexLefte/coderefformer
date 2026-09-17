import cv2
import os.path as osp
import random
import os
import numpy as np
import pickle
import torch
import torchvision.transforms as T

from .base_dataset import (BaseImageDataset, list_images as _list_images,
                           load_image_rgb as _load_image_rgb,
                           normalize_clip as _normalize)
from basicsr.utils import img2tensor
from utils.data_utils import apply_degradation_kernel_video


class UnpairedFolderImageDataset(BaseImageDataset):
    def __init__(self, data_opt, **kwargs):
        """ Folder dataset with unpaired data, used for generic restoration -> no conditional images
        """
        super(UnpairedFolderImageDataset, self).__init__(data_opt, **kwargs)

        # Get IDs 
        if osp.isfile(self.gt_dir):
            # Read IDs from the split's text file
            with open(self.gt_dir, 'r') as f:
                self.id_keys = [line.strip() for line in f.readlines()]
            self.gt_dir = osp.dirname(self.gt_dir)
        else:
            self.id_keys = sorted(os.listdir(self.gt_dir))

        # Get image paths per ID
        self.id_to_images = {}
        for id_key in self.id_keys:
            id_path = osp.join(self.gt_dir, id_key)
            if osp.isdir(id_path):
                # all images for this identity (supports multiple extensions)
                img_list = sorted([
                    osp.join(id_path, f)
                    for f in os.listdir(id_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ])
                if img_list:  # avoid empty dirs
                    self.id_to_images[id_key] = img_list
            else:
                # if gt_dir contains images directly, not folders per ID
                if id_key.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.id_to_images[id_key] = [osp.join(self.gt_dir, id_key)]

        # Reference conditioning
        self.conditional = data_opt.get("conditional", False)
        self.load_reference = data_opt.get("load_reference", False)
        self.num_references = data_opt.get("num_references", False)
        self.ref_augment = data_opt.get("ref_augment", False)
        self.ref_conds_path = data_opt.get("ref_conds_path", None)

        # Cached id embeddings
        self.id_embeddings_path = data_opt.get("id_embeddings_path", None)
        self._emb_tensor = None 

        # Cached FaRL token embeddings (optional)
        self.farl_embeddings_path = data_opt.get("farl_embeddings_path", None)
        self._farl_tensor      = None
        self._farl_name_to_idx = None

        # Facial components path
        self.facial_components_path = data_opt.get('facial_components_path', None)
        if self.facial_components_path is not None:
            self.facial_components = torch.load(self.facial_components_path, 
                                               weights_only=False, map_location="cpu")
        else:
            self.facial_components = None

        # GT Latent - for CodeFormer especially
        self.latent_gt_path = data_opt.get('latent_gt_path', None)
        if self.latent_gt_path is not None:
            self.load_latent_gt = True            
            self.latent_gt_dict = torch.load(self.latent_gt_path, weights_only=False)
        else:
            self.load_latent_gt = False 

        # Setup augmentation for reference images if needed
        if self.ref_augment:
            self.ref_augmentation = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomPerspective(distortion_scale=0.2, p=1.0, interpolation=T.InterpolationMode.BILINEAR),
            ])

        # Other attributes
        self.has_lr = False  # unpaired dataset does not have LR images
  
    def __len__(self):
        return len(self.id_keys)

    def __getitem__(self, id_idx):
        key = self.id_keys[id_idx]
        images = self.id_to_images[key]

        # Pick a random LR frame idx
        gt_idx = random.randint(0, len(images) - 1)
        gt_img_path = images[gt_idx]

        # get frames 
        gt = self._load_image(gt_img_path)

        # Load conditional reference images if needed
        if self.load_reference:
            id_key = self.id_keys[id_idx]

            # Select imposed number of references
            ref_img_paths = [p for p in images if p != gt_img_path]
            num_refs_available = len(ref_img_paths)
            ref_idx = self._select_references(num_refs_available)

            # Load only selected reference images
            ref_images = []
            ref_names = []
            if self.conditional:  # No reason to load entire references if not conditional
                # Load each reference image
                for i in ref_idx:
                    ref_img = self._load_image(ref_img_paths[i])
                    if self.ref_augment:
                        ref_img = self.ref_augmentation(ref_img)

                    ref_images.append(ref_img)
                    ref_names.append(osp.basename(ref_img_paths[i])[:-4])

                # Stack references
                if len(ref_images) > 0:
                    ref_images = torch.stack(ref_images, dim=0)
                else:
                    ref_images = torch.empty(0, 3, gt.shape[1], gt.shape[2])

            # Load only selected cached conditions
            if self.ref_conds_path is not None:
                cond_path = osp.join(self.ref_conds_path, f"{id_key}.pt")
                cond = torch.load(cond_path, map_location="cpu")
                cond_feats = cond["features"]   # [scale][ref]

                ref_conditions = []
                for scale_feats in cond_feats:
                    ref_conditions.append([scale_feats[i] for i in ref_idx])
            else:
                ref_conditions = []
        else:
            ref_images = torch.empty(0, 3, gt.shape[1], gt.shape[2])
            ref_conditions = []

        # Load ID embeddings
        id_emb = {}
        if self.id_embeddings_path is not None:
            # Get embedding for GT
            gt_emb = self._load_id_embeddings(gt_img_path)

            # Get embeddings for references
            id_emb = {
                'gt': gt_emb
            }
            if self.load_reference and ref_img_paths is not None:
                ref_emb_list = []
                for i in ref_idx:
                    ref_path = ref_img_paths[i]
                    ref_emb_list.append(self._load_id_embeddings(ref_path))
                if len(ref_emb_list) > 0:
                    ref_embs = torch.stack(ref_emb_list)
                    id_emb['refs'] = ref_embs

        # Load facial components/landmarks
        facial_components = { }
        if self.facial_components is not None:
            gt_lm_key = self._img_path_to_lm_key(gt_img_path)
            if gt_lm_key in self.facial_components:
                facial_components['gt'] = self.facial_components[gt_lm_key]
            else:
                facial_components['gt'] = None

            if self.load_reference and self.facial_components is not None:
                ref_lms = []
                for i in ref_idx:
                    ref_path = ref_img_paths[i]
                    ref_key = self._img_path_to_lm_key(ref_path)
                    if ref_key in self.facial_components:
                        ref_lms.append(self.facial_components[ref_key])
                    else:
                        ref_lms.append(None)
                facial_components['refs'] = ref_lms

        # Load GT & Ref latents
        if self.load_latent_gt:  # Load for GT
            latent_gt = self.latent_gt_dict['orig'][osp.basename(gt_img_path)[:-4]]
            if self.conditional:  # Load for Ref
                latent_refs = [self.latent_gt_dict['orig'][ref_name] for ref_name in ref_names]
                latent_refs = torch.tensor(latent_refs)

        return_dict = {
            'gt': gt, 
            'ref': ref_images,  # [N_refs, C, H, W]
            'ref_conditions': ref_conditions,
            'id_embeddings': id_emb,
            'facial_components': facial_components,
        }

        # Append gt latents
        if self.load_latent_gt:
            return_dict['latent_gt'] = latent_gt
            if self.conditional:
                return_dict['latent_ref'] = latent_refs

        # Append FaRL tokens
        if self.farl_embeddings_path is not None:
            farl_gt = self._load_farl_embeddings(gt_img_path)
            farl_dict = {'gt': farl_gt}
            if self.load_reference and ref_img_paths:
                farl_refs = [self._load_farl_embeddings(ref_img_paths[i]) for i in ref_idx]
                farl_dict['refs'] = torch.stack(farl_refs)
            return_dict['farl_embeddings'] = farl_dict

        # chw|rgb|float32
        return return_dict

    def _load_image(self, path):
        # Read image
        img = cv2.imread(os.path.join(self.gt_dir, path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        
        # Convert to float32 tensor, BGR to RGB, and normalize to [0, 1]
        img = img2tensor(img.astype(np.float32), bgr2rgb=True, float32=True) / 255.
        return img

    def _load_id_embeddings(self, img_path):
        if self.id_embeddings_path is None:
            return {}
        
        # load tensor into RAM on first access
        if self._emb_tensor is None:
            data = torch.load(self.id_embeddings_path, map_location="cpu")
            self._emb_tensor = data['embeddings']
            self._img_names = data['img_names']

            self._name_to_idx = {
                name: i for i, name in enumerate(self._img_names)
            }
        
        img_name = osp.basename(img_path)
        if img_name not in self._name_to_idx:
            return torch.zeros_like(self._emb_tensor[0])  # sau NaN

        idx = self._name_to_idx[img_name]
        return self._emb_tensor[idx]

    def _load_farl_embeddings(self, img_path):
        """Return (197, 768) FaRL tokens for one image. None if not configured."""
        if self.farl_embeddings_path is None:
            return None
 
        # Lazy-load on first access
        if self._farl_tensor is None:
            data = torch.load(self.farl_embeddings_path, map_location='cpu')
            self._farl_tensor      = data['farl_embeddings']   # (N, 197, 768)
            self._farl_name_to_idx = {
                name: i for i, name in enumerate(data['img_names'])
            }
 
        img_name = osp.basename(img_path)
        idx = self._farl_name_to_idx.get(img_name, None)
        if idx is None:
            return torch.zeros(197, 768, dtype=torch.float32)
 
        return self._farl_tensor[idx]   # (197, 768)

    def _img_path_to_lm_key(self, img_path):
        name = osp.basename(img_path)
        key = int(osp.splitext(name)[0])
        key = f'{key:08d}'
        return key

    def _select_references(self, num_available):
        k = self.num_references

        if num_available >= k:
            return random.sample(range(num_available), k)
        else:
            idxs = list(range(num_available))
            while len(idxs) < k:
                idxs.append(random.choice(idxs))
            return idxs



class UnpairedFolderVideoDataset(torch.utils.data.Dataset):
    """Training dataset: GT video frames loaded from folders, LQ generated on-the-fly.
 
    Degradation pipeline (apply_degradation_kernel_video):
        1. Blur   — kernel sampled once per clip, consistent across all frames
        2. Resize — scale sampled once, H/W forced to even for yuv420p
        3. Noise  — per-frame Gaussian or Poisson
        4. CRF    — codec + CRF sampled once per clip, encoded via PyAV
 
    Expected folder structure:
 
        <gt_dir>/
            <person_id>/
                images/                     <- HQ portrait images (references)
                    img_001.png
                videos/
                    video_1/
                        000000.png
                    video_2/
                        ...
                embeddings/
                    images/                 <- arcface.npy, farl.npy, index.pkl
                    videos/
                        video_1/            <- arcface.npy, index.pkl
                        video_2/
 
    One training sample = a randomly-selected window of `tempo_extent` consecutive
    frames from a randomly-selected video of a randomly-selected person.
    __len__ returns the number of persons so each person is seen once per epoch.
 
    Returns dict (normalisation to [-1, 1] is done inside __getitem__):
        'gt'               : (T, C, H, W)  float32 [-1, 1]
        'lr'               : (T, C, H, W)  float32 [-1, 1]
        'id_embeddings'    : {'gt': (T, 512), 'refs': (N, 512)}
        'farl_embeddings'  : {'refs': (N, 197, 768)}
        'ref'              : (N, C, H, W)  float32 [-1, 1]
        'identity_id'      : str
 
    Args:
        data_opt (dict):
            gt_dir              (str)  — root of the GT tree
            tempo_extent        (int)  — temporal window length (default: 5)
            num_references      (int)  — portrait images per sample (default: 3)
            ref_augment         (bool) — random color/geometry augmentation on refs (default: False)
            use_hflip           (bool) — random hflip on GT/LQ clip (default: True)
            first_degradation   (dict) — required degradation config
            second_degradation  (dict) — optional second degradation stage
    """
 
    def __init__(self, data_opt: dict, **kwargs):
        super().__init__()
 
        self.gt_dir             = data_opt['gt_dir']
        self.tempo_extent       = data_opt.get('tempo_extent', 5)
        self.num_references     = data_opt.get('num_references', 3)
        self.ref_augment        = data_opt.get('ref_augment', False)
        self.first_degradation  = data_opt.get('first_degradation', {})
        self.second_degradation = data_opt.get('second_degradation', None)
 
        # Reference augmentation — identical to UnpairedFolderImageDataset, no flip
        if self.ref_augment:
            REF_AUG_PROB = 0.7   # probability of augmenting each individual reference
            COLOR_PROB   = 0.5   # probability of color jitter
            AFFINE_PROB  = 0.5   # probability of random affine
            PERSP_PROB   = 0.3   # probability of random perspective
            if self.ref_augment:
                self.ref_aug_prob     = REF_AUG_PROB
                self.ref_augmentation = T.Compose([
                    T.RandomApply([
                        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                    ], p=COLOR_PROB),
                    T.RandomApply([
                        T.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05), interpolation=T.InterpolationMode.BILINEAR),
                    ], p=AFFINE_PROB),
                    T.RandomApply([
                        T.RandomPerspective(distortion_scale=0.2, p=1.0, interpolation=T.InterpolationMode.BILINEAR),
                    ], p=PERSP_PROB),
                ])

        # ── Discover person IDs ───────────────────────────────────────────────
        self.id_keys = sorted(
            d for d in os.listdir(self.gt_dir)
            if osp.isdir(osp.join(self.gt_dir, d))
        )
 
        # person_id -> List[image_path]  (HQ portrait images)
        self.id_to_images: dict = {}
        for iid in self.id_keys:
            img_dir = osp.join(self.gt_dir, iid, 'images')
            self.id_to_images[iid] = (
                _list_images(img_dir) if osp.isdir(img_dir) else []
            )
 
        # person_id -> {video_name: List[frame_path]}
        # Videos live directly under gt_dir/<person_id>/videos/
        self.id_to_videos: dict = {}
        for iid in self.id_keys:
            vid_root = osp.join(self.gt_dir, iid, 'videos')
            videos = {}
            if osp.isdir(vid_root):
                for v in sorted(os.listdir(vid_root)):
                    v_path = osp.join(vid_root, v)
                    if osp.isdir(v_path):
                        frames = _list_images(v_path)
                        if frames:
                            videos[v] = frames
            self.id_to_videos[iid] = videos
 
        # Keep only persons that have at least one video
        self.id_keys = [
            iid for iid in self.id_keys if self.id_to_videos.get(iid)
        ]
 
        # Lazy embedding caches — loaded once per (person, video) on first access
        self._img_emb_cache: dict = {}
        self._vid_emb_cache: dict = {}
 
        print(
            f'[UnpairedFolderVideoDataset] '
            f'{len(self.id_keys)} persons | '
            f'tempo_extent={self.tempo_extent}'
        )
 
    def __len__(self) -> int:
        # One entry per person — video and window are sampled randomly inside
        # __getitem__ so each epoch sees every person once.
        return len(self.id_keys)
 
    # ── Embedding loaders ─────────────────────────────────────────────────────
 
    def _load_image_emb(self, iid: str):
        """Load image ArcFace + FaRL embeddings for one identity (lazy cache)."""
        if iid in self._img_emb_cache:
            return self._img_emb_cache[iid]
 
        base = osp.join(self.gt_dir, iid, 'embeddings', 'images')
        if not osp.isdir(base):
            self._img_emb_cache[iid] = None
            return None
 
        data = {
            'arc':  np.load(osp.join(base, 'arcface.npy'), mmap_mode='r'),
            'farl': np.load(osp.join(base, 'farl.npy'),    mmap_mode='r'),
        }
        with open(osp.join(base, 'index.pkl'), 'rb') as f:
            data['index'] = pickle.load(f)
 
        self._img_emb_cache[iid] = data
        return data
 
    def _load_video_emb(self, iid: str, video_name: str):
        """Load video ArcFace embeddings for one clip (lazy cache)."""
        key = (iid, video_name)
        if key in self._vid_emb_cache:
            return self._vid_emb_cache[key]
 
        base = osp.join(self.gt_dir, iid, 'embeddings', 'videos', video_name)
        if not osp.isdir(base):
            self._vid_emb_cache[key] = None
            return None
 
        data = {
            'arc': np.load(osp.join(base, 'arcface.npy'), mmap_mode='r'),
        }
        with open(osp.join(base, 'index.pkl'), 'rb') as f:
            data['index'] = pickle.load(f)
 
        self._vid_emb_cache[key] = data
        return data
 
    # ── Reference selection ───────────────────────────────────────────────────
 
    def _select_references(self, n_available: int):
        """Sample self.num_references indices from [0, n_available).
 
        Repeats with replacement when the pool is smaller than requested.
        """
        if n_available == 0:
            return []
        if n_available < self.num_references:
            return random.choices(range(n_available), k=self.num_references)
        return random.sample(range(n_available), self.num_references)
 
    # ── __getitem__ ───────────────────────────────────────────────────────────
 
    def __getitem__(self, idx: int) -> dict:
        iid = self.id_keys[idx]
 
        # ── Select video and temporal window ──────────────────────────────────
        video_dict  = self.id_to_videos[iid]
        video_name  = random.choice(list(video_dict.keys()))
        frame_paths = video_dict[video_name]
 
        if len(frame_paths) < self.tempo_extent:
            raise RuntimeError(
                f'Clip too short: {iid}/{video_name} '
                f'({len(frame_paths)} < {self.tempo_extent})'
            )
 
        start      = random.randint(0, len(frame_paths) - self.tempo_extent)
        clip_paths = frame_paths[start: start + self.tempo_extent]
 
        # ── Load GT frames and synthesise LR on GPU ───────────────────────────
        gt_gpu = torch.stack([_load_image_rgb(p) for p in clip_paths]).cuda()
        # (T, C, H, W) float32 [0, 1]
 
        # apply_degradation_kernel_video expects (B, T, C, H, W)
        lr_gpu = apply_degradation_kernel_video(
            gt_gpu.unsqueeze(0),
            self.first_degradation,
        ).squeeze(0)  # (T, C, H, W)
 
        if self.second_degradation is not None:
            lr_gpu = apply_degradation_kernel_video(
                lr_gpu.unsqueeze(0),
                self.second_degradation,
            ).squeeze(0)
 
        # Normalise to [-1, 1] and move to CPU
        gt = _normalize(gt_gpu).cpu()
        lr = _normalize(lr_gpu).cpu()
 
        # ── GT ArcFace embeddings ─────────────────────────────────────────────
        vid_emb = self._load_video_emb(iid, video_name)
        gt_embs = []
        for p in clip_paths:
            stem = osp.splitext(osp.basename(p))[0]
            if vid_emb is not None:
                i = vid_emb['index'].get(stem, None)
                emb = (torch.from_numpy(vid_emb['arc'][i]).float()
                       if i is not None else torch.zeros(512))
            else:
                emb = torch.zeros(512)
            gt_embs.append(emb)
        gt_emb = torch.stack(gt_embs)  # (T, 512)
 
        # ── Reference images + embeddings ─────────────────────────────────────
        images  = self.id_to_images[iid]
        ref_idx = self._select_references(len(images))
        img_emb = self._load_image_emb(iid)
 
        ref_images, ref_arc, ref_farl = [], [], []
        for i in ref_idx:
            img = _load_image_rgb(images[i])
 
            # Apply augmentation to reference image if enabled (no flip — identical
            # to UnpairedFolderImageDataset; flip would break identity consistency)
            if self.ref_augment and random.random() < self.ref_aug_prob:
                img = self.ref_augmentation(img)

            ref_images.append(img)
 
            stem = osp.splitext(osp.basename(images[i]))[0]
            if img_emb is not None:
                j = img_emb['index'].get(stem, None)
                if j is not None:
                    ref_arc.append(
                        torch.from_numpy(img_emb['arc'][j]).float())
                    ref_farl.append(
                        torch.from_numpy(img_emb['farl'][j]).float())
 
        k = self.num_references
        ref_clip   = (_normalize(torch.stack(ref_images))
                      if ref_images else torch.zeros(k, 3, 512, 512))
        ref_arc_t  = (torch.stack(ref_arc)
                      if ref_arc  else torch.zeros(k, 512))
        ref_farl_t = (torch.stack(ref_farl)
                      if ref_farl else torch.zeros(k, 197, 768))
 
        return {
            'gt': gt,  # (T, C, H, W) [-1, 1]
            'lr': lr,  # (T, C, H, W) [-1, 1]
 
            'id_embeddings': {
                'gt':   gt_emb,     # (T, 512)
                'refs': ref_arc_t,  # (N, 512)
            },
            'farl_embeddings': {
                'refs': ref_farl_t,  # (N, 197, 768)
            },
 
            'ref':         ref_clip,  # (N, C, H, W) [-1, 1]
            'identity_id': iid,
        }
 