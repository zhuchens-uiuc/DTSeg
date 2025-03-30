import numpy as np
import cv2
import pickle
import os
from os.path import join as opj
import argparse
import glob
from tqdm import tqdm
from pathlib import Path
import scipy.io as scio

# import rasterio
# from rasterio.windows import Window
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from skimage import io
from PIL import Image
import math 


####
def _prepare_patching(img, window_size, mask_size, return_src_top_corner=False):
    """Prepare patch information for tile processing.
    
    Args:
        img: original input image
        window_size: input patch size
        mask_size: output patch size
        return_src_top_corner: whether to return coordiante information for top left corner of img
        
    """

    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step) * step_size
        return int(last_step), int(nr_step)

    im_h = img.shape[0]
    im_w = img.shape[1]
    # print(im_h, im_w)

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)


    # generating subpatches index from orginal
    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)

    return img, patch_info


class HuBMAPDataset(Dataset):
    def __init__(self, mode, filename, config):
        super().__init__()
        self.path = filename
        state = mode.split('/')[0]
        # self.data = rasterio.open(path)
        # if self.data.count != 3:
        #     subdatasets = self.data.subdatasets
        #     self.layers = []
        #     if len(subdatasets) > 0:
        #         for i,subdataset in enumerate(subdatasets,0):
        #             self.layers.append(rasterio.open(subdataset))
        self.data = np.array(Image.open(self.path))

        self.image_name = filename.split('/')[-1].split('.')[0]
        self.image_size = config.tile_size

        self.img_data, self.patch_info = _prepare_patching(
            self.data, config.tile_size, config.predict_size, True
        )
        
    def __len__(self):
        return len(self.patch_info)
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        coord_y, coord_x, row_idx, col_idx = self.patch_info[idx]

        img_patch = self.img_data[
                    coord_y : coord_y + self.image_size,
                    coord_x : coord_x + self.image_size,
                    ]

        return img_patch, row_idx, col_idx
    
    
    
    
def my_collate_fn(batch):
    img = []
    mask = []
    for sample in batch:
        img.append(sample['img'][None])
        mask.append(sample['mask'][None,:,:,None])
    img  = np.vstack(img)
    mask = np.vstack(mask)
    return {'img':img, 'mask':mask}

# python utils/img2tile_he_ihc.py --INPUT_PATH /data114_3/shaozc/Dataset/BCI_dataset/HE/
# python utils/img2tile_he_ihc.py --INPUT_PATH /data114_3/shaozc/Dataset/BCI_dataset/IHC/

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_PATH', default='/data114_3/shaozc/Dataset/natcom/live_dead2/', type=str) #(887, 878)
    parser.add_argument('--tile_size', default = 256, type=int)
    parser.add_argument('--predict_size', default = 192, type=int) #overlap
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_parse()
    train_val_filename = glob.glob(opj(args.INPUT_PATH, 'images', '*.png'))
    test_filename = glob.glob(opj(args.INPUT_PATH, 'masks', '*.png'))

    ignore_list = [
    ]
    train_val_filename = list(set(train_val_filename)-set(ignore_list))
    test_filename = list(set(test_filename)-set(ignore_list))

    Path(opj(args.INPUT_PATH, f'images_split_{args.tile_size}_{args.predict_size}')).mkdir(exist_ok=True, parents=True)
    Path(opj(args.INPUT_PATH, f'masks_split_{args.tile_size}_{args.predict_size}')).mkdir(exist_ok=True, parents=True)


    for filename in tqdm(train_val_filename):
        # filename = '/data114_2/shaozc/CellHE/MoNuSAC/MoNuSAC_images_and_annotations/TCGA-J4-A67T-01Z-00-DX1/TCGA-J4-A67T-01Z-00-DX1-3.tif'
        train_val_dataset = HuBMAPDataset(mode = 'images', filename = filename, config = args)
        train_val_dataloader = DataLoader(train_val_dataset, batch_size=8, num_workers=8)
        for i, (img_patch, row_idx, col_idx) in enumerate(train_val_dataloader):
            image_name = filename.split('/')[-1].split('.')[0]

            for j in range(len(img_patch)):
                cv2.imwrite(opj(args.INPUT_PATH, f'images_split_{args.tile_size}_{args.predict_size}', f'{image_name}_{int(row_idx[j])}_{int(col_idx[j])}_.png'), img_patch[j].numpy())
    

    for filename in tqdm(test_filename):
        test_dataset = HuBMAPDataset(mode = 'masks', filename = filename, config = args)
        test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=8)
        for i, (img_patch, row_idx, col_idx) in enumerate(test_dataloader):
            image_name = filename.split('/')[-1].split('.')[0]
            for j in range(len(img_patch)):
                cv2.imwrite(opj(args.INPUT_PATH, f'masks_split_{args.tile_size}_{args.predict_size}', f'{image_name}_{int(row_idx[j])}_{int(col_idx[j])}_.png'), img_patch[j].numpy())
    