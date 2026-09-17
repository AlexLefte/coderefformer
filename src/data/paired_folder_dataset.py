import cv2
import os.path as osp
import random
import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F

from .base_dataset import (BaseImageDataset, list_images as _list_images,
                           load_image_rgb as _load_image_rgb,
                           normalize_clip as _normalize)
from basicsr.utils import img2tensor


class BasePairedImageDataset(BaseImageDataset):
    def __init__(self, data_opt, **kwargs):
        super().__init__(data_opt, **kwargs)

        self.gt_dir = data_opt["gt_dir"]
        self.lr_dir = data_opt["lr_dir"]
        assert self.lr_dir is not None
        self.has_lr = True

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
            self.facial_components = torch.load(
                self.facial_components_path,
                weights_only=False,
                map_location="cpu"
            )
        else:
            self.facial_components = None
    
    def _gt_to_lr_path(self, gt_path):
        rel = osp.relpath(gt_path, self.gt_dir)
        lr_path = osp.join(self.lr_dir, rel)
        return lr_path

    def _lr_to_gt_path(self, lr_path):
        rel_path = osp.relpath(lr_path, self.lr_dir)  # get path relative to LR folder
        gt_path = osp.join(self.gt_dir, rel_path)     # join with GT base folder
        if not osp.exists(gt_path):
            raise FileNotFoundError(f"GT not found for LR {lr_path}: {gt_path}")
        return gt_path

    def _load_image(self, root, path):
        # Read image
        img = cv2.imread(osp.join(root, path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        
        # Convert to float32 tensor, BGR to RGB, and normalize to [0, 1]
        img = img2tensor(img.astype(np.float32), bgr2rgb=True, float32=True) / 255.
        return img
    
    def _img_path_to_lm_key(self, img_path):
        name = osp.basename(img_path)
        key = int(osp.splitext(name)[0])
        return f"{key:08d}"

    def _load_facial_components(self, img_path):
        if self.facial_components is None:
            return None
        key = self._img_path_to_lm_key(img_path)
        return self.facial_components.get(key, None)
    
    def _get_embedding(self, img_path):
        img_name = osp.basename(img_path)
        idx = self._name_to_idx.get(img_name, None)

        if idx is None:
            # fallback for missing images
            return torch.zeros_like(self._emb_tensor[0])

        return self._emb_tensor[idx]

    def _load_embeddings(self, gt_path, ref_paths=None):
        if self.id_embeddings_path is None:
            return None
        
        # load tensor into RAM on first access
        if self._emb_tensor is None:
            data = torch.load(self.id_embeddings_path, map_location="cpu")
            self._emb_tensor = data['embeddings']
            self._img_names = data['img_names']
            self._name_to_idx = {
                name: i for i, name in enumerate(self._img_names)
            }

        # GT embedding
        out = {
            "gt": self._get_embedding(gt_path)
        }

        # Reference embeddings
        if ref_paths is not None and len(ref_paths) > 0:
            ref_embs = [self._get_embedding(p) for p in ref_paths]
            out["refs"] = torch.stack(ref_embs)

        return out


class PairedFolderImageDataset(BasePairedImageDataset):
    def __init__(self, data_opt, **kwargs):
        super().__init__(data_opt, **kwargs)

        # ID discovery
        if osp.isfile(self.gt_dir):
            with open(self.gt_dir, 'r') as f:
                self.id_keys = [line.strip() for line in f.readlines()]
            self.gt_dir = osp.dirname(self.gt_dir)
        else:
            self.id_keys = sorted(os.listdir(self.gt_dir))

        # Build mapping: identity -> list of images
        self.id_to_images = {}
        for k in self.id_keys:
            p = osp.join(self.gt_dir, k)
            if osp.isdir(p):
                imgs = sorted([
                    osp.join(p, f) for f in os.listdir(p)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                self.id_to_images[k] = imgs

        # Build flat list: each entry is (identity, LR_path, list_of_all_HQ_images_for_this_id)
        self.samples = []
        for id_key, hq_images in self.id_to_images.items():
            # Only keep LR images that exist
            valid_lr_paths = [self._gt_to_lr_path(gt) for gt in hq_images if osp.exists(self._gt_to_lr_path(gt))]
            for lr_path in valid_lr_paths:
                self.samples.append((id_key, lr_path, hq_images))

        # reference config
        self.load_reference = data_opt.get("load_reference", False)
        self.num_references = data_opt.get("num_references", 0 if kwargs['train'] else -1)  # All references if testing
        self.ref_conds_path = data_opt.get("ref_conds_path", None)

    def __len__(self):
        return len(self.samples)

    def _select_references(self, num_available):
        k = self.num_references if self.num_references != -1 else num_available

        if num_available > k:
            # TODO: change back to random sampling after testing
            # return random.sample(range(num_available), k)
            return range(k)
        elif num_available == k:
            return range(num_available)
        else:
            idxs = list(range(num_available))
            while len(idxs) < k:
                idxs.append(random.choice(idxs))
            return idxs

    def _load_farl_embedding(self, img_path):
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

    def __getitem__(self, idx):
        id_key, lr_path, hq_images = self.samples[idx]

        # Get GT path from LR
        gt_path = self._lr_to_gt_path(lr_path)

        # Load LR and GT
        lr = self._load_image(self.lr_dir, osp.relpath(lr_path, self.lr_dir))
        gt = self._load_image(self.gt_dir, osp.relpath(gt_path, self.gt_dir))

        # ---- Load conditional reference images ----
        ref_images = []
        ref_paths = []
        ref_conditions = None
        if self.load_reference:
            ref_paths = [p for p in hq_images if p != gt_path]
            ref_idx = self._select_references(len(ref_paths))
            ref_paths = [ref_paths[i] for i in ref_idx]

            # Read reference images
            for p in ref_paths:
                ref_img = self._load_image(self.gt_dir, osp.relpath(p, self.gt_dir))
                ref_images.append(ref_img)

            if len(ref_images) > 0:
                ref_images = torch.stack(ref_images, dim=0)
            else:
                ref_images = torch.empty(0, 3, gt.shape[1], gt.shape[2])

            # load conditional embeddings
            if self.ref_conds_path is not None:
                cond = torch.load(osp.join(self.ref_conds_path, f"{id_key}.pt"), map_location="cpu")["features"]
                ref_conditions = []
                for scale_feats in cond:
                    ref_conditions.append([scale_feats[i] for i in ref_idx])

        # ---- Load ID embeddings ----
        id_embeddings = self._load_embeddings(
            gt_path=gt_path,
            ref_paths=ref_paths if self.load_reference else None
        )

        # ---- Facial components ----
        if getattr(self, "facial_components", False):
            facial_components = {"gt": self._load_facial_components(gt_path)}
            if self.load_reference:
                facial_components["refs"] = [self._load_facial_components(p) for p in ref_paths]
        else:
            facial_components = None

        # ---- Return dict ----
        return_dict = {
            "lr": lr,
            "gt": gt,
            "seq_idx": lr_path,
            "img_name": lr_path,
            "ref": ref_images,
        }

        if ref_conditions:
            return_dict["ref_conditions"] = ref_conditions
        if id_embeddings:
            return_dict["id_embeddings"] = id_embeddings
        if facial_components:
            return_dict["facial_components"] = facial_components

        # ---- FaRL tokens ----
        if self.farl_embeddings_path is not None:
            farl_gt = self._load_farl_embedding(gt_path)
            farl_dict = {'gt': farl_gt}
            if self.load_reference and ref_paths:
                farl_refs = [self._load_farl_embedding(p) for p in ref_paths]
                farl_dict['refs'] = torch.stack(farl_refs)
            return_dict['farl_embeddings'] = farl_dict

        return return_dict
    


class PairedFolderVideoDataset(torch.utils.data.Dataset):
    """Validation / test dataset: paired GT + LR video clips from folders.
 
    Loads the complete frame sequence for each clip — no windowing, no
    degradation, no augmentation.  Designed for batch_size=1.
 
    Expected folder structure on disk:
 
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
 
        <lr_dir>/
            <person_id>/
                videos/
                    video_1/
                        000000.png
                    video_2/
                        ...
 
    GT and LR frames are aligned by filename stem.  Stems present on only one
    side are silently skipped (handles encoder boundary drift in LR clips).
 
    Returns dict (one video per __getitem__ call):
        'gt'               : (T, C, H, W)  float32 [-1, 1]
        'lr'               : (T, C, H, W)  float32 [-1, 1]
        'ref'              : (N, C, H, W)  float32 [-1, 1]
        'id_embeddings'    : {'gt': (T, 512), 'refs': (N, 512)}
        'farl_embeddings'  : {'refs': (N, 197, 768)}
        'seq_idx'          : str  "<person_id>/<video_name>"
        'frame_names'      : List[str]  sorted frame stems
 
    Args:
        data_opt (dict):
            gt_dir         (str)  — root of the GT tree
            lr_dir         (str)  — root of the LR tree
            num_references (int)  — reference images per clip; -1 = all
                                    (default: 3)
            load_reference (bool) — load HQ reference images (default: True)
    """
 
    def __init__(self, data_opt: dict, **kwargs):
        super().__init__()
 
        self.gt_dir         = data_opt['gt_dir']
        self.lr_dir         = data_opt['lr_dir']
        self.num_references = data_opt.get('num_references', 3)
        self.load_reference = data_opt.get('load_reference', True)
 
        # ── Discover person IDs from GT root ──────────────────────────────────
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
 
        # ── Build flat clip list ──────────────────────────────────────────────
        # Each entry: (person_id, video_name, [gt_frame_paths], [lr_frame_paths])
        self.clips = []
        for iid in self.id_keys:
            gt_vid_root = osp.join(self.gt_dir, iid, 'videos')
            lr_vid_root = osp.join(self.lr_dir, iid, 'videos')
 
            if not osp.isdir(gt_vid_root) or not osp.isdir(lr_vid_root):
                continue
 
            for v in sorted(os.listdir(gt_vid_root)):
                gt_v = osp.join(gt_vid_root, v)
                lr_v = osp.join(lr_vid_root, v)
 
                if not osp.isdir(gt_v) or not osp.isdir(lr_v):
                    continue
 
                # Align frames by stem — skip any stem missing on either side
                gt_by_stem = {
                    osp.splitext(osp.basename(p))[0]: p
                    for p in _list_images(gt_v)
                }
                lr_by_stem = {
                    osp.splitext(osp.basename(p))[0]: p
                    for p in _list_images(lr_v)
                }
 
                common = sorted(set(gt_by_stem) & set(lr_by_stem))
                if not common:
                    continue
 
                self.clips.append((
                    iid, v,
                    [gt_by_stem[s] for s in common],
                    [lr_by_stem[s] for s in common],
                ))
 
        # Lazy embedding caches: loaded once per (person, video) on first access
        self._img_emb_cache: dict = {}
        self._vid_emb_cache: dict = {}
 
        print(
            f'[PairedFolderVideoDataset] '
            f'{len(self.id_keys)} persons | {len(self.clips)} clips'
        )
 
    def __len__(self) -> int:
        return len(self.clips)
 
    # ── Embedding loaders ─────────────────────────────────────────────────────
 
    def _load_image_emb(self, iid: str):
        """Load image ArcFace + FaRL embeddings for one identity (lazy)."""
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
        """Load video ArcFace embeddings for one clip (lazy)."""
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
 
        -1 means use all available; repeats with replacement when pool is small.
        """
        if n_available == 0:
            return []
        k = n_available if self.num_references == -1 else self.num_references
        if n_available < k:
            return random.choices(range(n_available), k=k)
        return random.sample(range(n_available), k)
 
    # ── __getitem__ ───────────────────────────────────────────────────────────
 
    def __getitem__(self, idx: int) -> dict:
        iid, video_name, gt_paths, lr_paths = self.clips[idx]
 
        frame_names = [osp.splitext(osp.basename(p))[0] for p in gt_paths]
 
        # Load and normalise complete clip sequences
        gt = _normalize(torch.stack([_load_image_rgb(p) for p in gt_paths]))
        lr = _normalize(torch.stack([_load_image_rgb(p) for p in lr_paths]))
        # Both: (T, C, H, W) float32 [-1, 1]
 
        # ── GT ArcFace embeddings ─────────────────────────────────────────────
        vid_emb = self._load_video_emb(iid, video_name)
        gt_embs = []
        for p in gt_paths:
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
            ref_images.append(_load_image_rgb(images[i]))
            stem = osp.splitext(osp.basename(images[i]))[0]
            if img_emb is not None:
                j = img_emb['index'].get(stem, None)
                if j is not None:
                    ref_arc.append(
                        torch.from_numpy(img_emb['arc'][j]).float())
                    ref_farl.append(
                        torch.from_numpy(img_emb['farl'][j]).float())
 
        k = len(ref_idx) if ref_idx else self.num_references
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
 
            'ref':         ref_clip,    # (N, C, H, W) [-1, 1]
            'seq_idx':     f'{iid}/{video_name}',
            'identity_id': iid,
            'video_name':  video_name,
            'frame_names': frame_names,
        }