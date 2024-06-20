import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image, ImageOps

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


class LoftrDataset(Dataset):
    def __init__(self,
                 image_dir,
                 pair_path,
                 img_size,
                 **kwargs):
        """
        Doppelgangers test dataset: loading images and loftr matches for Doppelgangers model.
        
        Args:
            image_dir (str): root directory for images.
            loftr_match_dir (str): root directory for loftr matches.
            pair_path (str): pair_list.npy path. This contains image pair information.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
        """
        super().__init__()
        self.image_dir = image_dir    
        self.pairs_info = np.load(pair_path, allow_pickle=True)
        self.img_size = img_size


    def __len__(self):
        return len(self.pairs_info)

    def __getitem__(self, idx):
        df = 8
        padding = True

        name0, name1, _, _, _ = self.pairs_info[idx]
        img0_pth = osp.join(self.image_dir, name0)
        img1_pth = osp.join(self.image_dir, name1)
        img0, mask0 = self.read_image(img0_pth, self.img_size, df, padding)
        img1, mask1 = self.read_image(img1_pth, self.img_size, df, padding)
        data = {
            'idx': idx,
            'image0': img0,
            'image1': img1,
            'mask0': mask0,
            'mask1':mask1
        }

        return data


    def read_image(self, img_pth, img_size, df, padding):
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





if __name__ == "__main__":
    pass