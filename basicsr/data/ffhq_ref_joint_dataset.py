import os

import cv2
import math
import random
import numpy as np
import os.path as osp
from scipy.io import loadmat
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, 
                                        adjust_hue, adjust_saturation, normalize)
from basicsr.data import gaussian_kernels as gaussian_kernels
from basicsr.data.transforms import augment
from basicsr.data.data_util import paths_from_folder
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class FFHQRefJointDataset(data.Dataset):

    def __init__(self, opt):
        super(FFHQRefJointDataset, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        assert self.gt_size >= self.in_size, 'Wrong setting.'
        
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.component_path = opt.get('component_path', None)
        self.latent_gt_path = opt.get('latent_gt_path', None)

        if self.component_path is not None:
            self.crop_components = True
            self.components_dict = torch.load(self.component_path)
            self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1.4)
            self.nose_enlarge_ratio = opt.get('nose_enlarge_ratio', 1.1)
            self.mouth_enlarge_ratio = opt.get('mouth_enlarge_ratio', 1.3)
        else:
            self.crop_components = False

        if self.latent_gt_path is not None:
            self.load_latent_gt = True            
            self.latent_gt_dict = torch.load(self.latent_gt_path)
        else:
            self.load_latent_gt = False  

        ## Data retrieval ##
        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError('lmdb backend is not supported for FFHQRefJointDataset, please use folder backend.')
            # self.io_backend_opt['db_paths'] = self.gt_folder
            # if not self.gt_folder.endswith('.lmdb'):
            #     raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            # with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
            #     self.paths = [line.split('.')[0] for line in fin]
        else:
            # self.paths = paths_from_folder(self.gt_folder)  # Alex L: get ID pairs instead
            # Get IDs 
            if osp.isfile(self.gt_folder):
                # Read IDs from the split's text file
                with open(self.gt_folder, 'r') as f:
                    self.id_keys = [line.strip() for line in f.readlines()]
                self.gt_folder = osp.dirname(self.gt_folder)
            else:
                self.id_keys = sorted(os.listdir(self.gt_folder)) 
            
            # Get image paths per ID
            self.id_to_images = {}
            for id_key in self.id_keys:
                id_path = osp.join(self.gt_folder, id_key)
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
                        self.id_to_images[id_key] = [osp.join(self.gt_folder, id_key)]

        # Reference image selection settings #
        self.conditional = opt.get("conditional", False)
        self.load_reference = opt.get("load_reference", False)
        self.num_references = opt.get("num_references", False)
        self.ref_conds_path = opt.get("ref_conds_path", None)
        
        # Cached id embeddings
        self.id_embeddings_path = opt.get("id_embeddings_path", None)
        self._emb_tensor = None 

        # Facial components path
        self.facial_components_path = opt.get('facial_components_path', None)
        if self.facial_components_path is not None:
            self.facial_components = torch.load(self.facial_components_path, 
                                               weights_only=False, map_location="cpu")
        else:
            self.facial_components = None

        ## Degradation settings ##
        # perform corrupt
        self.use_corrupt = opt.get('use_corrupt', True)
        self.use_motion_kernel = False
        # self.use_motion_kernel = opt.get('use_motion_kernel', True)

        if self.use_motion_kernel:
            self.motion_kernel_prob = opt.get('motion_kernel_prob', 0.001)
            motion_kernel_path = opt.get('motion_kernel_path', 'basicsr/data/motion-blur-kernels-32.pth')
            self.motion_kernels = torch.load(motion_kernel_path)

        if self.use_corrupt:
            # degradation configurations
            self.blur_kernel_size = self.opt['blur_kernel_size']
            self.kernel_list = self.opt['kernel_list']
            self.kernel_prob = self.opt['kernel_prob']
            # Small degradation
            self.blur_sigma = self.opt['blur_sigma']
            self.downsample_range = self.opt['downsample_range']
            self.noise_range = self.opt['noise_range']
            self.jpeg_range = self.opt['jpeg_range']
            # Large degradation
            self.blur_sigma_large = self.opt['blur_sigma_large']
            self.downsample_range_large = self.opt['downsample_range_large']
            self.noise_range_large = self.opt['noise_range_large']
            self.jpeg_range_large = self.opt['jpeg_range_large']

            # print
            logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
            logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
            logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
            logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob', None)
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob', None)
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')

        # to gray
        self.gray_prob = opt.get('gray_prob', 0.0)
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.

    def __len__(self):
        return len(self.id_keys)

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def _get_component_locations(self, name, status):
        components_bbox = self.components_dict[name]
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.gt_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.gt_size - components_bbox['right_eye'][0]
            components_bbox['nose'][0] = self.gt_size - components_bbox['nose'][0]
            components_bbox['mouth'][0] = self.gt_size - components_bbox['mouth'][0]
        
        locations_gt = {}
        locations_in = {}
        for part in ['left_eye', 'right_eye', 'nose', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            elif part == 'nose':
                half_len *= self.nose_enlarge_ratio
            elif part == 'mouth':
                half_len *= self.mouth_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations_gt[part] = loc
            loc_in = loc/(self.gt_size//self.in_size)
            locations_in[part] = loc_in
        return locations_gt, locations_in

    def _load_embedding(self, img_path):
        if self.id_embeddings_path is None:
            return {}
        
        # load tensor into RAM on first access
        if self._emb_tensor is None:
            self._emb_tensor = torch.load(self.id_embeddings_path, 
                                          map_location="cpu")['embeddings']

        idx = int(osp.splitext(osp.basename(img_path))[0])
        emb_data = self._emb_tensor[idx]
        return emb_data

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

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Select a random gt image for the desired identity
        key = self.id_keys[index]
        images = self.id_to_images[key]
        gt_idx = random.randint(0, len(images) - 1)
        gt_path = images[gt_idx]

        # load gt image
        name = osp.basename(gt_path)[:-4]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # load reference images
        if self.load_reference:
            # Select imposed number of references
            ref_img_paths = [p for p in images if p != gt_path]
            num_refs_available = len(ref_img_paths)
            ref_idx = self._select_references(num_refs_available)

            # Load only selected reference images
            img_ref = []
            ref_names = []
            if self.conditional:  # No reason to load entire references if not conditional
                # Load each reference image
                for i in ref_idx:
                    ref_path = ref_img_paths[i]
                    ref_name = osp.basename(ref_path)[:-4]
                    img_bytes = self.file_client.get(ref_path)
                    ref_img = imfrombytes(img_bytes, float32=True)
                    ref_img = img2tensor(ref_img, bgr2rgb=True, float32=True)
                    
                    # Append to list
                    img_ref.append(ref_img)
                    ref_names.append(ref_name)

                # Stack references
                if len(img_ref) > 0:
                    img_ref = torch.stack(img_ref, dim=0)
                else:
                    img_ref = torch.empty(0, 3, img_gt.shape[1], img_gt.shape[2])

            # Load only selected cached conditions
            if self.ref_conds_path is not None:
                cond_path = osp.join(self.ref_conds_path, f"{key}.pt")
                cond = torch.load(cond_path, map_location="cpu")
                cond_feats = cond["features"]   # [scale][ref]

                ref_conditions = []
                for scale_feats in cond_feats:
                    ref_conditions.append([scale_feats[i] for i in ref_idx])
            else:
                ref_conditions = []
        else:
            img_ref = torch.empty(0, 3, img_gt.shape[1], img_gt.shape[2])
            ref_conditions = []

        # Load ID embeddings
        id_emb = {}
        if self.id_embeddings_path is not None:
            # Get embedding for GT
            gt_emb = self._load_embedding(gt_path)

            # Get embeddings for references
            id_emb = {
                'gt': gt_emb
            }
            if self.load_reference and ref_img_paths is not None:
                ref_emb_list = []
                for i in ref_idx:
                    ref_path = ref_img_paths[i]
                    ref_emb_list.append(self._load_embedding(ref_path))
                ref_embs = torch.stack(ref_emb_list)
                id_emb['refs'] = ref_embs

        # Load facial components/landmarks
        facial_components = { }
        if self.facial_components is not None:
            gt_lm_key = self._img_path_to_lm_key(gt_path)
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

        # Augmentation -> hflip disabled for this datasets 
        img_gt, status = augment(img_gt, hflip=False, rotation=False, return_status=True)

        # Load latents
        if self.load_latent_gt:  # Load for GT
            latent_gt = self.latent_gt_dict['orig'][name]
        if self.conditional:  # Load for Ref
            latent_refs = [self.latent_gt_dict['orig'][ref_name] for ref_name in ref_names]
            latent_refs = torch.tensor(latent_refs)

        # Crop ROI elements/components if needed
        if self.crop_components:
            locations_gt, locations_in = self._get_component_locations(name, status)

        # generate in image
        img_in = img_gt
        if self.use_corrupt:
            # motion blur
            if self.use_motion_kernel and random.random() < self.motion_kernel_prob:
                m_i = random.randint(0,31)
                k = self.motion_kernels[f'{m_i:02d}']
                img_in = cv2.filter2D(img_in,-1,k)
            
            # gaussian blur
            kernel = gaussian_kernels.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, 
                [-math.pi, math.pi],
                noise_range=None)
            img_in = cv2.filter2D(img_in, -1, kernel)

            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_in = cv2.resize(img_in, (int(self.gt_size // scale), int(self.gt_size // scale)), interpolation=cv2.INTER_LINEAR)

            # noise
            if self.noise_range is not None:
                noise_sigma = np.random.uniform(self.noise_range[0] / 255., self.noise_range[1] / 255.)
                noise = np.float32(np.random.randn(*(img_in.shape))) * noise_sigma
                img_in = img_in + noise
                img_in = np.clip(img_in, 0, 1)

            # jpeg
            if self.jpeg_range is not None:
                jpeg_p = np.random.uniform(self.jpeg_range[0], self.jpeg_range[1])
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_p)]
                _, encimg = cv2.imencode('.jpg', img_in * 255., encode_param)
                img_in = np.float32(cv2.imdecode(encimg, 1)) / 255.

            # resize to in_size
            img_in = cv2.resize(img_in, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)


        # generate in_large with large degradation
        img_in_large = img_gt

        if self.use_corrupt:
            # motion blur
            if self.use_motion_kernel and random.random() < self.motion_kernel_prob:
                m_i = random.randint(0,31)
                k = self.motion_kernels[f'{m_i:02d}']
                img_in_large = cv2.filter2D(img_in_large,-1,k)
            
            # gaussian blur
            kernel = gaussian_kernels.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma_large,
                self.blur_sigma_large, 
                [-math.pi, math.pi],
                noise_range=None)
            img_in_large = cv2.filter2D(img_in_large, -1, kernel)

            # downsample
            scale = np.random.uniform(self.downsample_range_large[0], self.downsample_range_large[1])
            img_in_large = cv2.resize(img_in_large, (int(self.gt_size // scale), int(self.gt_size // scale)), interpolation=cv2.INTER_LINEAR)

            # noise
            if self.noise_range_large is not None:
                noise_sigma = np.random.uniform(self.noise_range_large[0] / 255., self.noise_range_large[1] / 255.)
                noise = np.float32(np.random.randn(*(img_in_large.shape))) * noise_sigma
                img_in_large = img_in_large + noise
                img_in_large = np.clip(img_in_large, 0, 1)

            # jpeg
            if self.jpeg_range_large is not None:
                jpeg_p = np.random.uniform(self.jpeg_range_large[0], self.jpeg_range_large[1])
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_p)]
                _, encimg = cv2.imencode('.jpg', img_in_large * 255., encode_param)
                img_in_large = np.float32(cv2.imdecode(encimg, 1)) / 255.

            # resize to in_size
            img_in_large = cv2.resize(img_in_large, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_in = self.color_jitter(img_in, self.color_jitter_shift)
            img_in_large = self.color_jitter(img_in_large, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            img_in = np.tile(img_in[:, :, None], [1, 1, 3])
            img_in_large = cv2.cvtColor(img_in_large, cv2.COLOR_BGR2GRAY)
            img_in_large = np.tile(img_in_large[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_large, img_gt = img2tensor([img_in, img_in_large, img_gt], bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_in = self.color_jitter_pt(img_in, brightness, contrast, saturation, hue)
            img_in_large = self.color_jitter_pt(img_in_large, brightness, contrast, saturation, hue)

        # round and clip
        img_in = np.clip((img_in * 255.0).round(), 0, 255) / 255.
        img_in_large = np.clip((img_in_large * 255.0).round(), 0, 255) / 255.

        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(img_in, self.mean, self.std, inplace=True)
        normalize(img_in_large, self.mean, self.std, inplace=True)
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_ref, self.mean, self.std, inplace=True)

        return_dict = {'in': img_in, 'in_large_de': img_in_large, 'gt': img_gt, 'gt_path': gt_path}

        if self.crop_components:
            return_dict['locations_in'] = locations_in
            return_dict['locations_gt'] = locations_gt

        if self.load_latent_gt:
            return_dict['latent_gt'] = latent_gt
            if self.conditional:
                return_dict['latent_ref'] = latent_refs

        if self.conditional:
            return_dict['in_ref'] = img_ref

        if self.id_embeddings_path:
            return_dict['id_embeddings'] = id_emb

        return return_dict