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
        self.data = io.imread(self.path)

        
        self.sz = config.tile_size
        if self.data.shape[0]<self.sz and self.data.shape[1]>self.sz:
            self.data = cv2.resize(self.data,(self.sz,self.data.shape[1]))
        elif self.data.shape[1]<self.sz and self.data.shape[0]>self.sz:
            self.data = cv2.resize(self.data,(self.data.shape[0],self.sz))
        elif self.data.shape[0]<self.sz and self.data.shape[1]<self.sz:
            self.data = cv2.resize(self.data,(self.sz,self.sz))

        self.h, self.w = self.data.shape[0], self.data.shape[1]

        self.num_h = self.h//self.sz+1
        self.num_w = self.w//self.sz+1
        if self.num_h>1:
            self.step_h = (self.h-self.sz)//(self.num_h-1)
        else: 
            self.step_h = 0
        if self.num_w>1:
            self.step_w = (self.w-self.sz)//(self.num_w-1)
        else: 
            self.step_w = 0    

        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio

        i_h = idx // self.num_w
        i_w = idx - i_h * self.num_w

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
        assert img_patch.shape[0]==self.sz, print(self.path)
        assert img_patch.shape[1]==self.sz, print(self.path)
        
        return img_patch, self.sz, px0, py0, self.step_w, self.step_h
    
    
    
    
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
    parser.add_argument('--INPUT_PATH', default='/data114_2/shaozc/CellHE/MoNuSAC/', type=str) #(887, 878)
    parser.add_argument('--tile_size', default = 256, type=int)
    parser.add_argument('--step_h', default = 210, type=int)
    parser.add_argument('--step_w', default = 207, type=int)
    parser.add_argument('--num_h', default = 4, type=int)
    parser.add_argument('--num_w', default = 4, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_parse()
    train_val_filename = glob.glob(opj(args.INPUT_PATH, 'MoNuSAC_images_and_annotations', '*', '*.tif'))
    test_filename = glob.glob(opj(args.INPUT_PATH, 'MoNuSAC_Testing_Data_and_Annotations', '*', '*.tif'))

    ignore_list = [
    ]
    train_val_filename = list(set(train_val_filename)-set(ignore_list))
    test_filename = list(set(test_filename)-set(ignore_list))

    Path(opj(args.INPUT_PATH, 'Train_split')).mkdir(exist_ok=True, parents=True)
    Path(opj(args.INPUT_PATH, 'Test_split')).mkdir(exist_ok=True, parents=True)
    Path(opj(args.INPUT_PATH, 'mask_split')).mkdir(exist_ok=True, parents=True)

    for filename in tqdm(train_val_filename):
        # filename = '/data114_2/shaozc/CellHE/MoNuSAC/MoNuSAC_images_and_annotations/TCGA-E2-A154-01Z-00-DX1/TCGA-E2-A154-01Z-00-DX1_6.tif'
        train_val_dataset = HuBMAPDataset(mode = 'Train/Images', filename = filename, config = args)
        train_val_dataloader = DataLoader(train_val_dataset, batch_size=8, num_workers=8)
        for i, (img_patch, tile_size, px0, py0, step_x, step_y) in enumerate(train_val_dataloader):
            image_name = filename.split('/')[-1].split('.')[0]
            if image_name[-5:] == '_mask':
                for j in range(len(img_patch)):
                    np.save(opj(args.INPUT_PATH, 'mask_split', f'{image_name[:-5]}_{tile_size[j]}_{px0[j]}_{py0[j]}_{step_x[j]}_{step_y[j]}_.npy'), img_patch[j].numpy())
            else:
                for j in range(len(img_patch)):
                    cv2.imwrite(opj(args.INPUT_PATH, 'Train_split', f'{image_name}_{tile_size[j]}_{px0[j]}_{py0[j]}_{step_x[j]}_{step_y[j]}_.png'), img_patch[j].numpy())
        

    for filename in tqdm(test_filename):
        test_dataset = HuBMAPDataset(mode = 'Test/Images', filename = filename, config = args)
        test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=8)
        for i, (img_patch, tile_size, px0, py0, step_x, step_y) in enumerate(test_dataloader):
            image_name = filename.split('/')[-1].split('.')[0]
            if image_name[-5:] == '_mask':
                for j in range(len(img_patch)):
                    np.save(opj(args.INPUT_PATH, 'mask_split', f'{image_name[:-5]}_{tile_size[j]}_{px0[j]}_{py0[j]}_{step_x[j]}_{step_y[j]}_.npy'), img_patch[j].numpy())
            else:
                for j in range(len(img_patch)):
                    cv2.imwrite(opj(args.INPUT_PATH, 'Test_split', f'{image_name}_{tile_size[j]}_{px0[j]}_{py0[j]}_{step_x[j]}_{step_y[j]}_.png'), img_patch[j].numpy())
        