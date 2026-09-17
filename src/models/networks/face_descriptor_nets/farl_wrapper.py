import torch
import torch.nn as nn
import torch.nn.functional as F


# CLIP ViT-B/16 (FaRL) normalisation constants.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def farl_preprocess_neg1to1(imgs, size=224):
    """Model-side FaRL preprocessing.

    Converts images in [-1, 1] to CLIP-normalised tensors of the given size
    using bilinear resize. Matches the online extraction path used during
    training in the CodeFormer(+VSR) models.

    Args:
        imgs: (B, 3, H, W) tensor in [-1, 1] on any device.
    Returns:
        (B, 3, size, size) CLIP-normalised tensor on imgs.device.
    """
    x = (imgs + 1.0) / 2.0
    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    mean = torch.tensor(_CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def farl_visual_tokens(visual, imgs):
    """Run a CLIP ViT-B/16 ``visual`` transformer keeping all 197 tokens.

    This is the token-extraction core shared by every FaRL call site (online
    model path, offline cache scripts, analysis scripts). Only the upstream
    preprocessing differs between callers.

    Args:
        visual: a CLIP visual transformer (e.g. ``net_FaRL.visual``).
        imgs:   (B, 3, 224, 224) already CLIP-normalised, on any device.
    Returns:
        (B, 197, 768) float tokens (CLS + 196 patch tokens) on imgs.device.
    """
    x = visual.conv1(imgs.to(visual.conv1.weight.dtype))            # (B, D, 14, 14)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)      # (B, 196, D)
    cls = (visual.class_embedding.to(x.dtype)
           .unsqueeze(0).unsqueeze(0)
           .expand(x.shape[0], -1, -1))                             # (B, 1, D)
    x = torch.cat([cls, x], dim=1)                                  # (B, 197, D)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)                                          # (197, B, D)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                                          # (B, 197, D)
    x = visual.ln_post(x).float()                                   # (B, 197, 768)
    return x


class FaRLEncoder(nn.Module):
    """FaRL wrapper that returns (197, 768) tokens, identical to the official
    CLIP forward used in FaRL."""

    def __init__(self, farl_path, device="cuda"):
        super().__init__()
        import clip  # lazy — only needed when a FaRLEncoder is constructed

        # load CLIP backbone
        self.model, self.preprocess = clip.load("ViT-B/16", device="cpu")

        state = torch.load(farl_path, map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)

        self.model.eval()
        self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device

    @torch.no_grad()
    def forward(self, img):
        """
        img: Tensor [3, H, W] or [B, 3, H, W] in [-1, 1]
        returns: (197, 768) or (B, 197, 768)
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img = farl_preprocess_neg1to1(img.to(self.device))
        x = farl_visual_tokens(self.model.visual, img)
        return x.squeeze(0).cpu()
