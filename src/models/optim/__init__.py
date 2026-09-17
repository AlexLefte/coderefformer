import torch.nn as nn
import torch.optim as optim


def define_criterion(criterion_opt):
    if criterion_opt is None:
        return None
    else:
        criterion_type = criterion_opt['type']

    # parse
    if criterion_type == 'MSE':
        criterion = nn.MSELoss(reduction=criterion_opt['reduction'])

    elif criterion_type == 'L1':
        criterion = nn.L1Loss(reduction=criterion_opt['reduction'])

    elif criterion_type == 'CB':
        from .losses import CharbonnierLoss
        criterion = CharbonnierLoss(reduction=criterion_opt['reduction'])

    elif criterion_type == 'CosineSimilarity':
        from .losses import CosineSimilarityLoss
        criterion = CosineSimilarityLoss()

    elif criterion_type == 'WeightedLPIPS':
        from .losses import WeightedLPIPSLoss
        criterion = WeightedLPIPSLoss(**criterion_opt)

    elif criterion_type == 'LPIPS':
        from .losses import LPIPSLoss
        criterion = LPIPSLoss(use_input_norm=criterion_opt['use_input_norm'],
                              range_norm=criterion_opt['use_range_norm'])

    elif criterion_type == 'GAN':
        from .losses import VanillaGANLoss
        criterion = VanillaGANLoss(reduction=criterion_opt['reduction'])

    elif criterion_type == 'WPGAN':
        from .losses import WGANLoss
        criterion = WGANLoss(lambda_gp=criterion_opt.get('lambda_gp', 10))

    elif criterion_type == 'WGAN_SoftPlus':
        from .losses import WGANSoftPlusLoss
        criterion = WGANSoftPlusLoss()

    elif criterion_type == 'HingeGAN':
        from .losses import HingeGANLoss
        criterion = HingeGANLoss()

    else:
        raise ValueError('Unrecognized criterion: {}'.format(
            criterion_type))

    return criterion


def define_lr_schedule(schedule_opt, optimizer):
    if schedule_opt is None:
        return None

    # parse
    if schedule_opt['type'] == 'FixedLR':
        schedule = None

    elif schedule_opt['type'] == 'MultiStepLR':
        schedule = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=schedule_opt['milestones'],
            gamma=schedule_opt['gamma']
        )

    elif schedule_opt['type'] == 'CosineAnnealingLR_Restart':
        from .lr_schedules import CosineAnnealingLR_Restart
        schedule = CosineAnnealingLR_Restart(
            optimizer, schedule_opt['periods'],
            eta_min=schedule_opt['eta_min'],
            restarts=schedule_opt['restarts'],
            weights=schedule_opt['restart_weights']
        )

    else:
        raise ValueError('Unrecognized lr schedule: {}'.format(
            schedule_opt['type']))

    return schedule