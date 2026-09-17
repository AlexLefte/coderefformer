import onnx
import torch
import torch.nn as nn
import onnx2torch

from pathlib import Path


class IdentityEncoder(nn.Module):
    """ArcFace wrapper to extract identity embeddings from face images.
    """
    def __init__(self, model_path, center_crop=True, resize_hw=(112, 112)):
        super().__init__()
        model_path = Path(model_path)
        if model_path.suffix == ".onnx":
            # model = onnx2torch.convert(model_path)
            model = onnx.load(model_path)
            model = onnx2torch.convert(model)
        elif model_path.suffix == ".pt":
            model = torch.load(model_path)
        else:
            raise NotImplementedError
        #  freeze_model(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.center_crop = center_crop
        self.resize_hw = resize_hw

    def preprocess(self, img):
        if self.center_crop:
            h, w = img.shape[-2:]
            img = img[:, :, int(h * 0.0625): int(h * 0.9375), int(w * 0.0625): int(w * 0.9375)]
        if self.resize_hw is not None:
            img = nn.functional.interpolate(img, self.resize_hw, mode='area')
        return img

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x) # B, Channels (512)
        return x