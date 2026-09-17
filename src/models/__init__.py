def define_model(opt):
    name = opt['model']['name'].lower()
    if name == 'coderefformer':
        from .codeformer_model import CodeFormerJointRefModel
        model = CodeFormerJointRefModel(opt)
    elif name == 'codeformervsr':
        from .codeformer_video_model import CodeFormerVSRModel
        model = CodeFormerVSRModel(opt)
    else:
        raise ValueError('Unrecognized model: {}'.format(opt['model']['name']))

    return model
