from torch.utils.data import DataLoader
from .unpaired_folder_dataset import UnpairedFolderImageDataset, UnpairedFolderVideoDataset
from .paired_folder_dataset import PairedFolderImageDataset, PairedFolderVideoDataset


def create_dataloader(opt, dataset_idx='train'):
    # setup params
    data_opt = opt['dataset'].get(dataset_idx)
    data_opt['gt_bit_depth'] = opt.get('gt_bit_depth', 8)
    video_task = data_opt.get('video', False)

    # -------------- loader for training -------------- #
    if dataset_idx == 'train':
        if data_opt.get('lr_dir', None) is not None:
            # create dataset
            if video_task:
                dataset = PairedFolderVideoDataset(
                    data_opt,
                    train=True)
            else:
                dataset = PairedFolderImageDataset(
                    data_opt,
                    train=True)
        else:
            # create dataset
            if video_task:
                dataset = UnpairedFolderVideoDataset(
                    data_opt,
                    train=True)
            else:
                dataset = UnpairedFolderImageDataset(
                    data_opt,
                    train=True)

        # create data loader
        loader = DataLoader(
            dataset=dataset,
            batch_size=data_opt['batch_size'],
            shuffle=True,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    # -------------- loader for testing -------------- #
    elif dataset_idx.startswith('test'):
        # paired dataset with provided LR sequences
        if video_task:
            dataset = PairedFolderVideoDataset(
                            data_opt,
                            train=False
                        )
        else:
            dataset = PairedFolderImageDataset(
                data_opt,
                train=False
            )

        # create data loader
        loader = DataLoader(
            dataset=dataset,
            batch_size=data_opt.get('batch_size', 1),
            shuffle=False,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    else:
        raise ValueError('Unrecognized dataset index: {}'.format(dataset_idx))

    return loader