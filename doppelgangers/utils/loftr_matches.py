import torch
import cv2
import numpy as np
import os.path as osp
import tqdm
from PIL import Image, ImageOps

from ..third_party.loftr import LoFTR, default_cfg
from ..datasets.loftr_dataset import LoftrDataset

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    if w_new == 0:
        w_new = df
    if h_new == 0:
        h_new = df
    return w_new, h_new


def read_image(img_pth, img_size, df, padding):
    if str(img_pth).endswith('gif'):
        
        pil_image = ImageOps.grayscale(Image.open(str(img_pth)))
        img_raw = np.array(pil_image)
    else:
        img_raw = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)

    w, h = img_raw.shape[1], img_raw.shape[0]
    w_new, h_new = get_resized_wh(w, h, img_size)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    if padding:  # padding
        pad_to = max(h_new, w_new)    
        mask = np.zeros((pad_to, pad_to), dtype=bool)
        mask[:h_new,:w_new] = True
        mask = mask[::8,::8]
    
    image = cv2.resize(img_raw, (w_new, h_new))
    pad_image = np.zeros((1, pad_to, pad_to), dtype=np.float32)
    pad_image[0,:h_new,:w_new]=image/255.

    return pad_image, mask


def save_loftr_matches(data_path, pair_path, output_path, cfg, model_weight_path="weights/outdoor_ds.ckpt"):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(model_weight_path)['state_dict'])
    matcher = matcher.eval().cuda()

    # initial dataset
    img_size = 1024
    te_dataset = LoftrDataset(image_dir=data_path, pair_path=pair_path, img_size=img_size)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=cfg.data.test.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers, drop_last=False
    )

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            batch = {k:v.cuda() for k,v in batch.items()}
            matcher(batch)
            bs = batch['image0'].shape[0]
            for b_id in range(bs):
                idx = batch['idx'][b_id].cpu().numpy()
                mask = batch['m_bids'] == b_id
                output_dir = osp.join(output_path, f'loftr_match/{idx}.npy')
                mkpts0 = batch['mkpts0_f'][mask].cpu().numpy()
                mkpts1 = batch['mkpts1_f'][mask].cpu().numpy()
                mconf = batch['mconf'][mask].cpu().numpy()
                np.save(output_dir, {"kpt0": mkpts0, "kpt1": mkpts1, "conf": mconf})


