import torch
import numpy as np
import os.path as osp
import tqdm
import argparse
import yaml
import os

import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from third_party.loftr import LoFTR, default_cfg
from datasets.loftr_dataset import LoftrDataset

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

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
        sampler=DistributedSampler(dataset)
    )

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

def main(rank: int, world_size: int, config_path: str, data_path:str, pair_path:str, output_path:str, model_weight_path:str):
    ddp_setup(rank, world_size)

    # Load model
    gpu_id = rank
    cfg = load_config(config_path)
    model = LoFTR(config=default_cfg)
    model_weight_path = "weights/outdoor_ds.ckpt" if model_weight_path is None else model_weight_path
    model.load_state_dict(torch.load(model_weight_path)['state_dict'])
    model = model.cuda(gpu_id)
    # model = DDP(model, device_ids=[gpu_id])
    model.eval()

    # Load data
    img_size = 1024
    dataset = LoftrDataset(image_dir=data_path, pair_path=pair_path, img_size=img_size)
    test_loader = prepare_dataloader(dataset, cfg.data.test.batch_size, 4)

    # Inference with LoFTR and get prediction
    output_path = os.path.join(output_path, 'loftr_match')
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            idx0 = batch['idx'][0].cpu().numpy()
            if os.path.isfile(os.path.join(output_path, f'{idx0}.npy')):
                continue
            batch = {k:v.cuda(gpu_id, non_blocking=True) for k,v in batch.items()}
            model(batch)
            bs = batch['image0'].shape[0]
            for b_id in range(bs):
                idx = batch['idx'][b_id].cpu().numpy()
                mask = batch['m_bids'] == b_id
                output_dir = osp.join(output_path, f'{idx}.npy')
                mkpts0 = batch['mkpts0_f'][mask].cpu().numpy()
                mkpts1 = batch['mkpts1_f'][mask].cpu().numpy()
                mconf = batch['mconf'][mask].cpu().numpy()
                np.save(output_dir, {"kpt0": mkpts0, "kpt1": mkpts1, "conf": mconf})

    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LOFTR DDP prediction')
    parser.add_argument('--config_path', type=str, required=True, help='The configuration file.')
    parser.add_argument('--model_weight_path', required=False, type=str, help='The configuration file.')
    parser.add_argument('--data_path', type=str, required=True, help='The configuration file.')
    parser.add_argument('--pair_path', type=str, required=True, help='The configuration file.')
    parser.add_argument('--output_path', type=str, required=True, help='The configuration file.')

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(
        main, 
        args=(world_size, args.config_path, args.data_path, args.pair_path, args.output_path, args.model_weight_path),
        nprocs=world_size,
        join=True
    )