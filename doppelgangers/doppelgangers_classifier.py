import torch
import numpy as np
import tqdm
import argparse
import yaml
import os

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os
import yaml
import torch
import argparse
import importlib
import torch.distributed
from torch.backends import cudnn
import tqdm
import numpy as np

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return config

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank: int, world_size: int, cfg: dict, model_weight_path: str):
    ddp_setup(rank, world_size)

    # Load model
    gpu_id = rank

    # basic setup
    cudnn.benchmark = True
    multi_gpu = False
    strict = True

    # initial dataset
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data)
    test_loader = loaders['test_loader_ddp']

    # initial model
    decoder_lib = importlib.import_module(cfg.models.decoder.type)
    decoder = decoder_lib.decoder(cfg.models.decoder)
    decoder = decoder.cuda(gpu_id)

    # load pretrained model
    ckpt = torch.load(model_weight_path)
    import copy
    new_ckpt = copy.deepcopy(ckpt['dec'])
    if not multi_gpu:
        for key, value in ckpt['dec'].items():
            if 'module.' in key:
                new_ckpt[key[len('module.'):]] = new_ckpt.pop(key)
    elif multi_gpu:
        for key, value in ckpt['dec'].items():                
            if 'module.' not in key:
                new_ckpt['module.'+key] = new_ckpt.pop(key)
    decoder.load_state_dict(new_ckpt, strict=strict)
    decoder.eval()
    decoder = DDP(decoder, device_ids=[gpu_id])

    # evaluate on test set
    pair_idx_list = list()
    gt_list = list()
    pred_list = list()
    prob_list = list()
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            data['image'] = data['image'].cuda(gpu_id)
            gt = data['gt'].cuda(gpu_id)
            score = decoder(data['image'])
            for i in range(score.shape[0]):
                pair_idx_list.append(data['idx'].cpu().numpy())
                prob_list.append(score[i].cpu().numpy())
                pred_list.append(torch.argmax(score, dim=1)[i].cpu().numpy())
                gt_list.append(gt[i].cpu().numpy())
    
    pair_idx_list = np.array(pair_idx_list).reshape(-1)
    gt_list = np.array(gt_list).reshape(-1)
    pred_list = np.array(pred_list).reshape(-1)
    prob_list = np.array(prob_list).reshape(-1, 2)    
    np.save(os.path.join(cfg.data.output_path, f"pair_probability_list_{gpu_id}.npy"), {'pair_idx': pair_idx_list, 'pred': pred_list, 'gt': gt_list, 'prob': prob_list})

    destroy_process_group()


def merge_results(output_path: str):
    # Load all files properly
    from glob import glob
    files = glob(f"{output_path}/pair_probability_list_*.npy")
    c_data = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in d.item().keys():
            if k in c_data:
                c_data[k].append( d.item().get(k) )
            else:
                c_data[k] = [ d.item().get(k) ]
    # Concatenate
    c_data = { k: np.concatenate(c_data[k], axis=0) for k in c_data.keys() }
    # Sort by pair idx
    pos = np.argsort( c_data['pair_idx'] )
    c_data = { k: v[pos] for k, v in c_data.items() }
    c_data.pop('pair_idx', None)
    # Save!
    np.save(os.path.join(cfg.data.output_path, f"pair_probability_list.npy"), c_data)
    # Delete previous files
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Doppelgangers Classificiation DDP prediction')
    parser.add_argument('--config_path', type=str, required=True, help='The configuration file.')
    parser.add_argument('--model_weight_path', required=False, type=str, help='The configuration file.')

    args = parser.parse_args()
    cfg = load_config(args.config_path)
    world_size = torch.cuda.device_count()
    # Start processes
    mp.spawn(
        main, 
        args=(world_size, cfg, args.model_weight_path),
        nprocs=world_size,
        join=True
    )

    # Join results into one file
    merge_results(cfg.data.output_path)
