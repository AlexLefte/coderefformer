import os
import os.path as osp
import random
import torch

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        # TODO: delete - just for debugging
        self.paths = self.paths[:50]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'in': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageRefDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        self.num_references = opt.get('num_references', 1)
        self.load_reference = opt.get('load_reference', True)

        # ----- build paired paths ----- #

        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError('lmdb backend is not supported for PairedImageRefDataset, please use folder backend.')
            # self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            # self.io_backend_opt['client_keys'] = ['lq', 'gt']
            # self.paths = paired_paths_from_lmdb(
            #     [self.lq_folder, self.gt_folder], ['lq', 'gt']
            # )
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            raise NotImplementedError('meta_info_file backend is not supported for PairedImageRefDataset, please use folder backend.')
            # self.paths = paired_paths_from_meta_info_file(
            #     [self.lq_folder, self.gt_folder],
            #     ['lq', 'gt'],
            #     self.opt['meta_info_file'],
            #     self.filename_tmpl
            # )
        else:
            # self.paths = paired_paths_from_folder(
            #     [self.lq_folder, self.gt_folder],
            #     ['lq', 'gt'],
            #     self.filename_tmpl
            # )
            if osp.isfile(self.gt_folder):
                # Read IDs from the split's text file
                with open(self.gt_folder, 'r') as f:
                    self.id_keys = [line.strip() for line in f.readlines()]
                self.gt_folder = osp.dirname(self.gt_folder)
            else:
                self.id_keys = sorted(os.listdir(self.gt_folder))

        # TODO: delete - just for debugging keep first 50 identities
        self.id_keys = self.id_keys[:100]

        # ----- BUILD ID MAPPING -----
        self.id_to_images = {}
        for k in self.id_keys:
            p = osp.join(self.gt_folder, k)
            if osp.isdir(p):
                imgs = sorted([
                    osp.join(p, f) for f in os.listdir(p)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                if len(imgs) > 1:  # TODO: redo, kept only minimum 2 images
                    self.id_to_images[k] = imgs
        
        # Build flat list: each entry is (identity, image_path, other_images_in_identity)
        self.paths = []
        for id_key, images in self.id_to_images.items():
            for img_path in images:

                rel_path = osp.relpath(img_path, self.gt_folder)
                lq_path = osp.join(self.lq_folder, rel_path)

                self.paths.append({
                    'gt_path': img_path,
                    'lq_path': lq_path,
                    'identity': id_key,
                    'ref_paths': images
                })

        # reference config
        self.conditional = opt.get("conditional", False)
        self.load_reference = opt.get("load_reference", False)
        self.num_references = opt.get("num_references", False)
        self.ref_conds_path = opt.get("ref_conds_path", None)
    
    def _select_references(self, num_available):
        k = self.num_references if self.num_references != -1 else num_available

        if num_available >= k:
            return random.sample(range(num_available), k)
        else:
            idxs = list(range(num_available))
            while len(idxs) < k:
                idxs.append(random.choice(idxs))
            return idxs
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'),
                **self.io_backend_opt
            )

        sample = self.paths[index]
        gt_path = sample['gt_path']
        lq_path = sample['lq_path']
        ref_paths = sample['ref_paths']

        # ----- load main pair -----

        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

        # ----- load references -----

        ref_imgs = []
        if self.load_reference:
            # Select imposed number of references
            ref_img_paths = [p for p in ref_paths if p != gt_path]
            print(f"For img: {gt_path} num refs: {len(ref_img_paths)}")
            num_refs_available = len(ref_img_paths)
            ref_indices = self._select_references(num_refs_available)

            # Read reference images
            for ref_idx in ref_indices:
                ref_path = ref_img_paths[ref_idx]
                ref_img = imfrombytes(
                    self.file_client.get(ref_path, 'gt'),
                    float32=True
                )
                ref_imgs.append(ref_img)

        # ----- to tensor -----

        img_gt, img_lq = img2tensor(
            [img_gt, img_lq],
            bgr2rgb=True,
            float32=True
        )

        if self.load_reference and len(ref_imgs) > 0:
            ref_imgs = img2tensor(ref_imgs, bgr2rgb=True, float32=True)
            ref_imgs = torch.stack(ref_imgs, dim=0)
        else:
            ref_imgs = torch.empty(0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            print(ref_imgs.shape)
            normalize(ref_imgs, self.mean, self.std, inplace=True)

        return {
            'in': img_lq,
            'gt': img_gt,
            'in_ref': ref_imgs,
            'lq_path': lq_path,
            'gt_path': gt_path
        }