from utils.net_utils import get_upsampling_func
import torch.nn as nn


def define_generator(opt):
    net_G_opt = opt['model']['generator']
    net_G_name = opt['model']['g_name'].lower()

    # CodeFormer networks
    if net_G_name == 'codeformer':
        from .codeformer_nets import CodeFormer
        net_G = CodeFormer(**net_G_opt)
    elif net_G_name == 'coderefformer_v1':
        # V1 covers the former V3 as well: set use_hq_transformer: true in the
        # generator config to enable the frozen HQ reference transformer.
        from .coderefformer_nets import CodeRefFormerV1
        net_G = CodeRefFormerV1(**net_G_opt)
    elif net_G_name == 'coderefformer_v5':
        from .coderefformer_nets import CodeRefFormerV5
        net_G = CodeRefFormerV5(**net_G_opt)

    # Video CodeFormer architectures
    elif net_G_name == 'singleframevideovsr':
        from .codeformer_video_nets import SingleFrameVideoVSR
        net_G = SingleFrameVideoVSR(**net_G_opt)
    elif net_G_name == 'codeformervsrv1':
        from .codeformer_video_nets import ConvGRUVideoVSR
        net_G = ConvGRUVideoVSR(**net_G_opt)
    elif net_G_name == 'temporalattnvideovsr':
        from .codeformer_video_nets import TemporalAttnVideoVSR
        net_G = TemporalAttnVideoVSR(**net_G_opt)

    else:
        raise ValueError('Unrecognized generator: {}'.format(
            net_G_name))

    return net_G


def define_discriminator(net_D_opt, net_D_name):
    if net_D_name == 'stylegan2discriminator':  # spatio-temporal discriminator
        from basicsr.archs.stylegan2_arch import StyleGAN2Discriminator
        net_D = StyleGAN2Discriminator(**net_D_opt)
    elif net_D_name == 'vqgandiscriminator':
        from .vqgan_arch import VQGANDiscriminator
        net_D = VQGANDiscriminator(**net_D_opt)
    elif net_D_name == 'vqgandiscriminator3d':
        from .vqgan_arch import VQGANDiscriminator3D
        net_D = VQGANDiscriminator3D(**net_D_opt)
    else:
        raise ValueError('Unrecognized discriminator: {}'.format(
            net_D_name))

    return net_D


def define_id_net(model_path):
    from models.networks.face_descriptor_nets.arcface_wrapper import IdentityEncoder
    net_ID = IdentityEncoder(model_path=model_path)

    return net_ID
