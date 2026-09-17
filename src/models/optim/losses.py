import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks.perceptual_nets.vgg_nets import VGGFeatureExtractor


def compute_gradient_penalty(D, real_samples, fake_samples, lambda_gp=10, device='cuda'):
        """
        Computes the gradient penalty for WGAN-GP.

        Args:
            D (nn.Module): The discriminator (or critic).
            real_samples (Tensor): Real images [B, C, H, W].
            fake_samples (Tensor): Fake images [B, C, H, W].
            device (str): Device to perform computation on.
            lambda_gp (float): Gradient penalty coefficient.

        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        batch_size = real_samples.size(0)

        # Interpolate between real and fake
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        # Forward pass
        d_interpolates, _ = D(interpolates)

        # If D returns [B, 1], flatten to [B]
        if d_interpolates.ndim > 1:
            d_interpolates = d_interpolates.view(-1)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)  # L2 norm

        gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return gp


class VanillaGANLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(VanillaGANLoss, self).__init__()
        self.crit = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, status, **kwargs):
        """
            :param status: boolean, True/False
        """
        target = torch.empty_like(input).fill_(int(status))
        loss = self.crit(input, target)
        return loss


class WGANLoss(nn.Module):
    def __init__(self, lambda_gp=10.0):
        super(WGANLoss, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, input, status, **kwargs):
        """
        :param input: Discriminator output
        :param status: 1 if real, 0 if fake
        :return: Wasserstein loss
        """
        # Real -> hope for high scores → maximize D(x_real) → -D(x_real)
        # Fake -> hope for low scores → minimize D(x_fake) → D(x_fake)
        return -input.mean() if status == 1 else input.mean()

class WGANSoftPlusLoss(nn.Module):
    def __init__(self):
        super(WGANSoftPlusLoss, self).__init__()

    def forward(self, input, status, **kwargs):
        return F.softplus(-input).mean() if status == 1 else F.softplus(input).mean()

class HingeGANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, status, **kwargs):
        is_critic_update = kwargs['is_critic_update']
        if is_critic_update:
            # Discriminator loss
            if status:
                # For real: max(0, 1 - pred)
                return F.relu(1.0 - input).mean()
            else:
                # For fake: max(0, 1 + pred)
                return F.relu(1.0 + input).mean()
        else:
            # Generator loss
            return -input.mean()

class CharbonnierLoss(nn.Module):
    """ Charbonnier Loss (robust L1)
    """

    def __init__(self, eps=1e-6, reduction='sum'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)

        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
        else:
            raise NotImplementedError
        return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        diff = F.cosine_similarity(input, target, dim=1, eps=self.eps)
        loss = 1.0 - diff.mean()

        return loss


class LPIPSLoss(nn.Module):
    def __init__(self, 
            use_input_norm=True,
            range_norm=False,):
        super(LPIPSLoss, self).__init__()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred   = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred   = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return lpips_loss.mean()
    

class WeightedLPIPSLoss(nn.Module):
    def __init__(self,
                 **kwargs):
        super().__init__()

        base_loss_type = kwargs.get('base_loss_type', 'L1')

        self.feature_layers = kwargs.get(
            'feature_layers',
            ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'])
        self.feature_weights = kwargs.get(
            'feature_weights',
            [0.1, 0.1, 1, 1, 1])

        self.feature_extractor = VGGFeatureExtractor(self.feature_layers, 
                                                     normalize=False)

        # Base loss
        self.reduction = kwargs.get('reduction', 'mean')
        if base_loss_type == 'L1':
            self.base_loss = nn.L1Loss(reduction=self.reduction)
        elif base_loss_type == 'MSE':
            self.base_loss = nn.MSELoss(reduction=self.reduction)
        elif base_loss_type == 'CB':
            from .losses import CharbonnierLoss
            self.base_loss = CharbonnierLoss(reduction=self.reduction)
        else:
            raise ValueError(f"Unknown base_loss_type {base_loss_type}")

    def forward(self, pred, target):

        pred_feats = self.feature_extractor(pred)
        with torch.no_grad():
            target_feats = self.feature_extractor(target)

        loss = 0.0
        for i, (pf, tf) in enumerate(zip(pred_feats, target_feats)):
            loss += self.base_loss(pf, tf.detach()) * self.feature_weights[i]

        return loss