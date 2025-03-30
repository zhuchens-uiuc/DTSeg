import numpy as np
import cv2
import pickle
import os
from os.path import join as opj
import argparse
import glob
from tqdm import tqdm
from pathlib import Path

# import rasterio
# from rasterio.windows import Window
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

    

class HuBMAPDataset(Dataset):
    def __init__(self, mode, filename, config):
        super().__init__()
        path = opj(config.INPUT_PATH,mode, filename)
        mask_path = opj(config.INPUT_PATH,'mask', filename.split('.')[0]+'.npy')
        # self.data = rasterio.open(path)
        # if self.data.count != 3:
        #     subdatasets = self.data.subdatasets
        #     self.layers = []
        #     if len(subdatasets) > 0:
        #         for i,subdataset in enumerate(subdatasets,0):
        #             self.layers.append(rasterio.open(subdataset))
        self.data = cv2.imread(path)
        self.h, self.w = self.data.shape[0], self.data.shape[1]
        self.sz = config.tile_size
        self.num_h = config.num_h
        self.num_w = config.num_w
        self.step_h = config.step_h
        self.step_w = config.step_w
        self.mask = np.load(mask_path)
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio

        i_h = idx // self.num_h
        i_w = idx % self.num_w

        y = i_h*self.step_h
        x = i_w*self.step_w

        py0,py1 = y, y+self.sz
        px0,px1 = x, x+self.sz
        
        # placeholder for input tile (before resize)
        # img_patch  = np.zeros((self.sz,self.sz,3), np.uint8)
        # mask_patch = np.zeros((self.sz,self.sz), np.uint8)
        
        # # replace the value for img patch
        # if self.data.count == 3:
        #     img_patch[0:py1-py0, 0:px1-px0] =\
        #         np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((py0,py1),(px0,px1))), 0,-1)
        # else:
        #     for i,layer in enumerate(self.layers):
        #         img_patch[0:py1-py0, 0:px1-px0, i] =\
        #             layer.read(1,window=Window.from_slices((py0,py1),(px0,px1)))
        
        img_patch = self.data[py0:py1, px0:px1]

        # replace the value for mask patch
        mask_patch = self.mask[py0:py1,px0:px1]
        
        return img_patch, mask_patch, self.sz, px0, py0, self.step_w, self.step_h
    
    
    
    
def my_collate_fn(batch):
    img = []
    mask = []
    for sample in batch:
        img.append(sample['img'][None])
        mask.append(sample['mask'][None,:,:,None])
    img  = np.vstack(img)
    mask = np.vstack(mask)
    return {'img':img, 'mask':mask}



def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_PATH', default='/data114_2/shaozc/LiveCell/images', type=str) #520*704
    parser.add_argument('--tile_size', default = 256, type=int)
    parser.add_argument('--step_h', default = 66, type=int)
    parser.add_argument('--step_w', default = 112, type=int)
    parser.add_argument('--num_h', default = 4, type=int)
    parser.add_argument('--num_w', default = 4, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_parse()
    train_val_filename = os.listdir(opj(args.INPUT_PATH, 'livecell_train_val_images'))
    test_filename = os.listdir(opj(args.INPUT_PATH, 'livecell_test_images'))

    Path(opj(args.INPUT_PATH, 'livecell_train_val_images_split')).mkdir(exist_ok=True, parents=True)
    Path(opj(args.INPUT_PATH, 'livecell_test_images_split')).mkdir(exist_ok=True, parents=True)
    Path(opj(args.INPUT_PATH, 'mask_split')).mkdir(exist_ok=True, parents=True)

    # for filename in tqdm(train_val_filename):
    #     train_val_dataset = HuBMAPDataset(mode = 'livecell_train_val_images', filename = filename, config = args)
    #     train_val_dataloader = DataLoader(train_val_dataset, batch_size=8, num_workers=8)
    #     for i, (img_patch, mask_patch, tile_size, px0, py0, step_x, step_y) in enumerate(train_val_dataloader):
    #         image_name = filename.split('.')[0]
    #         for j in range(len(img_patch)):
    #             cv2.imwrite(opj(args.INPUT_PATH, 'livecell_train_val_images_split', f'{image_name}_{tile_size[j]}_{px0[j]}_{py0[j]}_{step_x[j]}_{step_y[j]}_.png'), img_patch[j].numpy())
    #             np.save(opj(args.INPUT_PATH, 'mask_split', f'{image_name}_{tile_size[j]}_{px0[j]}_{py0[j]}_{step_x[j]}_{step_y[j]}_.npy'), mask_patch[j].numpy())

    for filename in tqdm(test_filename):
        test_dataset = HuBMAPDataset(mode = 'livecell_test_images', filename = filename, config = args)
        test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=8)
        for i, (img_patch, mask_patch, tile_size, px0, py0, step_x, step_y) in enumerate(test_dataloader):
            image_name = filename.split('.')[0]
            for j in range(len(img_patch)):
                cv2.imwrite(opj(args.INPUT_PATH, 'livecell_test_images_split', f'{image_name}_{tile_size[j]}_{px0[j]}_{py0[j]}_{step_x[j]}_{step_y[j]}_.png'), img_patch[j].numpy())
                np.save(opj(args.INPUT_PATH, 'mask_split', f'{image_name}_{tile_size[j]}_{px0[j]}_{py0[j]}_{step_x[j]}_{step_y[j]}_.npy'), mask_patch[j].numpy())
