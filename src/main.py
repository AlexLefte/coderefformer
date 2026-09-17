import os
import os.path as osp
import math
import argparse
import cv2
import yaml
import time
import torch
import numpy as np
import sys

from basicsr.utils import tensor2img
from collections import defaultdict
from tqdm import tqdm
from data import create_dataloader
from models import define_model
from models.networks import define_generator
from metrics.metric_calculator import MetricCalculator
from metrics.model_summary import register, profile_model
from utils import base_utils, data_utils
from torch.utils.tensorboard import SummaryWriter
import psutil
from torchvision.transforms.functional import normalize


def get_gpu_temp():
    import subprocess
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        return int(output.decode().strip())
    except Exception as e:
        print(f"Temp check failed: {e}")
        return -1


def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB")

    if memory_usage > 50000:
        print(f"Stopped training due to increased memory usage..")
        exit(1)


def train(opt):
    # logging
    logger = base_utils.get_logger('base')
    logger.info('{} Options {}'.format('='*20, '='*20))
    base_utils.print_options(opt, logger)

    # create a summary writer
    tb_logger = SummaryWriter(log_dir=os.path.join(opt['exp_dir'], 'tb_logger'))

    # create train data loader
    train_loader = create_dataloader(opt, dataset_idx='train')

    # create test data loaders
    test_loaders = {}
    for dataset_idx in sorted(opt['dataset'].keys()):
        if dataset_idx.startswith('test'):
            test_loaders[dataset_idx] = create_dataloader(opt, dataset_idx=dataset_idx)

    # create model
    model = define_model(opt)

    # training configs
    total_sample = len(train_loader.dataset)
    iter_per_epoch = len(train_loader)
    total_iter = opt['train']['total_iter']
    accum_steps = opt['train'].get('accum_steps', 1)
    total_epoch = int(math.ceil(total_iter * accum_steps / iter_per_epoch))
    start_iter, iter = opt['train']['start_iter'], 1
    gt_bit_depth = opt['train'].get('gt_bit_depth', 8)
    gt_dtype = np.uint8 if gt_bit_depth == 8  else np.uint16
    has_lr = opt['dataset']['train'].get('lr_dir') is not None
    is_video = opt['dataset']['train'].get('video', False)

    test_freq = opt['test']['test_freq']
    log_freq = opt['logger']['log_freq']
    ckpt_freq = opt['logger']['ckpt_freq']
    tb_freq = opt['logger']['tb_freq']
    logger.info('Number of training samples: {}'.format(total_sample))
    logger.info('Total epochs needed: {} for {} iterations'.format(
        total_epoch, total_iter))

    # Step accumulation  
    raw_iter = 0
    print(f"Accumulating gradients every: {accum_steps} steps..")
    
    # define metric calculator
    metric_calculator = MetricCalculator(opt)

    # train
    for epoch in range(total_epoch):
        print(f'Running epoch: {epoch}...') 
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Get batch time
            batch_start = time.time()

            # update iter
            raw_iter += 1
            if raw_iter % accum_steps == 0:
                iter += 1
                is_last_accum = True

                # update learning rate
                model.update_learning_rate()
            else: 
                is_last_accum = False
            curr_iter = start_iter + iter
            if iter > total_iter:
                logger.info('Finish training')
                break                

            # prepare data
            if not is_video:  # Apply degradation on GPU if processing images
                degraded_dict = data_utils.sample_degraded_image(opt, 
                                                                gt=data['gt'], 
                                                                lr=data.get('lr', None), 
                                                                ref=data.get('ref', None),
                                                                kernel=None, 
                                                                has_lr=has_lr)
                data.update(degraded_dict)

            # train for a mini-batch
            model.train(data, iter=iter, accum_steps=accum_steps, is_last_accum=is_last_accum)

            # Skip if not accumulating
            if not is_last_accum:
                continue

            # update running log
            model.update_running_log()

            # log
            if log_freq > 0 and iter % log_freq == 0:
                # basic info
                msg = '[epoch: {} | iter: {}'.format(epoch, curr_iter)
                for lr_type, lr in model.get_current_learning_rate().items():
                    msg += ' | {}: {:.2e}'.format(lr_type, lr)
                msg += '] '

                # loss info
                log_dict = model.get_running_log()
                msg += ', '.join([
                    '{}: {:.3e}'.format(k, v) for k, v in log_dict.items()])
                logger.info(msg)

            if tb_freq > 0 and iter % tb_freq == 0:
                # Log the items
                for key, value in log_dict.items():
                    tb_logger.add_scalar(f'train/{key}', value, curr_iter)

            # save model
            if ckpt_freq > 0 and iter % ckpt_freq == 0:
                model.save(curr_iter)

            # evaluate performance
            if test_freq > 0 and iter % test_freq == 0:
                # setup model index
                model_idx = f"G_{curr_iter:07d}"

                # dict to accumulate all sequences across datasets
                all_seq_metrics = defaultdict(list)

                # Enable eval mode
                if model.ema_decay == 0:
                    print(f"Swithing to eval mode")
                    model.net_G.eval()

                # for each testset
                for dataset_idx in sorted(opt['dataset'].keys()):
                    # use dataset with prefix `test`
                    if not dataset_idx.startswith('test'):
                        continue

                    ds_name = opt['dataset'][dataset_idx]['name']
                    logger.info(
                        'Testing on {}: {}'.format(dataset_idx, ds_name))

                    # create data loader
                    test_loader = test_loaders[dataset_idx]

                    # infer and compute metrics for each sequence
                    print(f'Running validation on {ds_name}...')
                    start_infer = time.time()
                    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):                        
                        # Pre-process inputs
                        pre_processed_dict = data_utils.pre_process_data(data)
                        data.update(pre_processed_dict)
                        seq_idx = data.get('seq_idx', [None])[0]
                        frm_idx = data.get('frm_idx', None)

                        # infer
                        output_dict = model.infer(data)  # thwc|rgb|uint8         
                        hr_data = output_dict['hr_data']

                        # save results (optional)
                        if opt['test']['save_res']:
                            res_dir = osp.join(
                                opt['test']['res_dir'], ds_name, model_idx)
                            
                            # Post-process for saving
                            hr_data_np = data_utils.post_process(hr_data, bit_depth=gt_bit_depth)

                            # Save according to type of media
                            if is_video:
                                res_seq_dir = osp.join(res_dir, seq_idx)
                                data_utils.save_sequence(
                                    res_seq_dir, hr_data_np, frm_idx, to_bgr=True, dtype=gt_dtype)
                            else:
                                # Save single image
                                os.makedirs(res_dir, exist_ok=True)
                                hr_names = data['img_name']
                                for hr_img, img_name in zip(hr_data_np, hr_names):     
                                    img_path = osp.join(res_dir, os.path.basename(img_name))

                                    # Convert RGB to BGR for cv2
                                    hr_bgr = hr_img[..., ::-1] if len(hr_img.shape) == 3 else hr_img
                                    cv2.imwrite(img_path, hr_bgr.astype(gt_dtype))

                        # compute metrics for the current sequence
                        metric_calculator.compute_sequence_metrics(
                            seq_idx, '', '', true_seq=pre_processed_dict['gt'], pred_seq=hr_data)
                    
                    stop_infer = time.time()
                    print(f'Elapsed validation time: {stop_infer - start_infer}s.')

                    # print directly and log to tb
                    print(f"Epoch {epoch+1}. Iteration {iter}.")
                    metric_calculator.display_results(iter, tb_logger, ds_name)
                    
                    # save/print metrics
                    if opt['test'].get('save_json'):
                        # save results to json file
                        json_path = osp.join(
                            opt['test']['json_dir'], '{}_avg.json'.format(ds_name))
                        metric_calculator.save_results(model_idx, json_path, override=True, iter=iter, tb_logger=tb_logger) 
                    
                    # accumulate per-sequence metrics for global average
                    for metric_type in metric_calculator.metric_opt.keys():
                        for seq_metrics in metric_calculator.metric_dict.values():
                            all_seq_metrics[metric_type].extend(seq_metrics[metric_type])

                    # Reset the metric calculator
                    metric_calculator.reset()    

                # compute global average over all test datasets
                global_avg = {metric: float(np.mean(values)) for metric, values in all_seq_metrics.items()}

                # log to TensorBoard
                if tb_logger is not None:
                    for metric, value in global_avg.items():
                        tb_logger.add_scalar(f'val/global/{metric}', value, iter)

                print("Global average across all test datasets:", global_avg)

            if curr_iter % 500 == 0:
                temp = get_gpu_temp()
                print(f"GPU temp: {temp}°C.")
                if temp > 80: 
                    logger.warning(f"High GPU temp: {temp}°C. Sleeping for 5min...")
                    time.sleep(300)

                # Log memory usage
                log_memory_usage()

    if tb_logger:
        tb_logger.close()


def test(opt):
    # logging
    logger = base_utils.get_logger('base')
    if opt['verbose']:
        logger.info('{} Configurations {}'.format('=' * 20, '=' * 20))
        base_utils.print_options(opt, logger)

    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # setup model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]
        
        # log
        logger.info('=' * 40)
        logger.info('Testing model: {}'.format(model_idx))
        logger.info('=' * 40)

        # create model
        opt['model']['g_load_path'] = load_path
        model = define_model(opt)

        # Get bit depth
        gt_bit_depth = opt.get('gt_bit_depth', 8)
        gt_dtype = np.uint8 if gt_bit_depth == 8  else np.uint16

        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # use dataset with prefix `test`
            if not dataset_idx.startswith('test'):
                continue

            ds_name = opt['dataset'][dataset_idx]['name']
            logger.info('Testing on {}: {}'.format(dataset_idx, ds_name))
            is_video = opt['dataset'][dataset_idx].get('video', False)

            # define metric calculator
            try:
                metric_calculator = MetricCalculator(opt)
            except Exception as e:
                print(f'Exception: {e}')

            # create data loader
            test_loader = create_dataloader(opt, dataset_idx=dataset_idx)

            # infer and store results for each sequence
            for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                # Pre-process inputs
                pre_processed_dict = data_utils.pre_process_data(data)
                data.update(pre_processed_dict)
                seq_idx = data.get('seq_idx', [None])[0]
                frm_idx = data.get('frm_idx', None)

                # infer
                output_dict = model.infer(data)  # thwc|rgb|uint8         
                hr_data = output_dict['hr_data']
       
                # save results (optional)
                if opt['test']['save_res']:
                    res_dir = osp.join(
                        opt['test']['res_dir'], ds_name, model_idx)
                    
                    # Post-process for saving
                    hr_data_np = data_utils.post_process(hr_data, bit_depth=gt_bit_depth)

                    # Save according to type of media
                    if is_video:
                        res_seq_dir = osp.join(res_dir, seq_idx)
                        data_utils.save_sequence(
                            res_seq_dir, hr_data_np, frm_idx, to_bgr=True, dtype=gt_dtype)
                    else:
                        # Save single image
                        os.makedirs(res_dir, exist_ok=True)
                        hr_names = data['img_name']
                        for hr_img, img_name in zip(hr_data_np, hr_names):     
                            img_path = osp.join(res_dir, os.path.basename(img_name))

                            # Convert RGB to BGR for cv2
                            hr_bgr = hr_img[..., ::-1] if len(hr_img.shape) == 3 else hr_img
                            cv2.imwrite(img_path, hr_bgr.astype(gt_dtype))

                # compute metrics for the current sequence
                try:
                    metric_calculator.compute_sequence_metrics(
                        seq_idx, '', '', true_seq=data['gt'], pred_seq=hr_data)
                except Exception as ex:
                    print(f"Error: {ex}.")

            # save/print metrics
            try:
                if opt['test'].get('save_json'):
                    # save results to json file
                    json_path = osp.join(
                        opt['test']['json_dir'], '{}_avg.json'.format(ds_name))
                    metric_calculator.save_results(model_idx, json_path, override=True)
                
                # Also print
                metric_calculator.display_results()
 
            except Exception as ex:
                print(f"Error: {ex}")

            # Reset the metric calculator
            metric_calculator.reset() 

            logger.info('-' * 40)

    # logging
    logger.info('Finish testing')
    logger.info('=' * 40)


def profile(opt, lr_size, test_speed=False):
    # logging
    logger = base_utils.get_logger('base')
    logger.info('{} Model Information {}'.format('='*20, '='*20))
    base_utils.print_options(opt['model']['generator'], logger)

    # basic configs
    scale = opt['scale']
    device = torch.device(opt['device'])

    # create model
    net_G = define_generator(opt).to(device)

    # get dummy input
    dummy_input_dict = net_G.generate_dummy_input(lr_size)
    for key in dummy_input_dict.keys():
        dummy_input_dict[key] = dummy_input_dict[key].to(device)

    # profile
    register(net_G, dummy_input_dict)
    gflops, params = profile_model(net_G)

    # Move back to cpu for testing purposes
    for key in dummy_input_dict.keys():
        dummy_input_dict[key] = dummy_input_dict[key].cpu()

    logger.info('-' * 40)
    logger.info('Super-resolute data from {}x{}x{} to {}x{}x{}'.format(
        *lr_size, lr_size[0], lr_size[1]*scale, lr_size[2]*scale))
    logger.info('Parameters (x10^6): {:.3f}'.format(params/1e6))
    logger.info('FLOPs (x10^9): {:.3f}'.format(gflops))
    logger.info('-' * 40)

    # test running speed
    if test_speed:
        n_test = 30
        tot_time = []
        cpu2gpu_times = []
        gpu2cpu_times = []
        fnet_times = []
        srnet_times = []
        for i in range(n_test):
            start_time = time.time()

            # Transfer CPU -> GPU
            for key in dummy_input_dict.keys():
                dummy_input_dict[key] = dummy_input_dict[key].to(device)
            # end_time = time.time()
            # cpu2gpu_times.append(end_time - start_time)

            # Inference
            with torch.no_grad():
                # _, fnet_time, srnet_time = net_G(**dummy_input_dict)
                _ = net_G(**dummy_input_dict)
                # fnet_times.append(fnet_time)
                # srnet_times.append(srnet_time)
            torch.cuda.synchronize()
            
            # Transfer GPU -> CPU
            start_time_2 = time.time()
            for key in dummy_input_dict.keys():
                dummy_input_dict[key] = dummy_input_dict[key].cpu()
            end_time = time.time()
            gpu2cpu_times.append(end_time - start_time_2)
            
            end_time = time.time()
            tot_time.append(end_time - start_time)
            # times.append(end_time - start_time)
        # print(f"CPU2GPU avg inf time: {sum(cpu2gpu_times[3:])/len(cpu2gpu_times[3:])}")
        # print(f"FNet avg inf time: {sum(fnet_times[3:])/len(fnet_times[3:])}")
        # print(f"SRNet avg inf time: {sum(srnet_times[3:])/len(srnet_times[3:])}")
        # print(f"GPU2CPU avg inf time: {sum(gpu2cpu_times[3:])/len(gpu2cpu_times[3:])}")
        logger.info('Speed (FPS): {:.3f} (averaged for {} runs)'.format(
            sum(tot_time[3:])/len(tot_time[3:]), n_test - 3))
        logger.info('-' * 40)


if __name__ == '__main__':
    # ----------------- parse arguments ----------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='directory of the current experiment')
    parser.add_argument('--mode', type=str, required=True,
                        help='which mode to use (train|test|profile)')
    parser.add_argument('--opt', type=str, default='config/train.yml',
                        help='path to the option yaml file')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU index, -1 for CPU')
    parser.add_argument('--lr_size', type=str, default='3x256x256',
                        help='size of the input frame')
    parser.add_argument('--test_speed', action='store_true',
                        help='whether to test the actual running speed')
    args = parser.parse_args()


    # ----------------- get options ----------------- #
    print(args.exp_dir)
    with open(args.opt, 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)
        opt['opt_path'] = args.opt

    # ----------------- general configs ----------------- #
    # experiment dir
    opt['exp_dir'] = os.path.join('experiments', args.exp_dir)

    # random seed
    base_utils.setup_random_seed(opt['manual_seed'])

    # logger
    base_utils.setup_logger('base')
    opt['verbose'] = opt.get('verbose', False)

    # device
    if args.gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            opt['device'] = 'cuda'
        else:
            opt['device'] = 'cpu'
    else:
        opt['device'] = 'cpu'


    # ----------------- train ----------------- #
    if args.mode == 'train':
        # setup paths
        base_utils.setup_paths(opt, mode='train')

        # run
        opt['is_train'] = True
        train(opt)

    # ----------------- test ----------------- #
    elif args.mode == 'test':
        # setup paths
        base_utils.setup_paths(opt, mode='test')

        # run
        opt['is_train'] = False
        test(opt)

    # ----------------- profile ----------------- #
    elif args.mode == 'profile':
        lr_size = tuple(map(int, args.lr_size.split('x')))

        # run
        profile(opt, lr_size, args.test_speed)

    else:
        raise ValueError(
            'Unrecognized mode: {} (train|test|profile)'.format(args.mode))
