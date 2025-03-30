import random
import torch
import pandas as pd
from pathlib import Path
import numpy as np

import cv2
import torch.utils.data as data
from torch.utils.data import dataloader
from torch.utils.data import Dataset
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


#https://github.com/Lewislou/cell-seg/blob/main/train_convnext_hover..py

#######data augmentation
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    LoadImaged,
    # LoadImaged_modified,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
    apply_transform,
)






class FeaturebaseaugData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.state = state
        self.csv_dir = self.dataset_cfg.label_dir
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)
        self.image_size = self.dataset_cfg.image_size
#https://albumentations.ai/docs/examples/example_kaggle_salt/

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train_image'].dropna()
            self.data = self.data[:int(len(self.data)*self.dataset_cfg.label_ratio)]
            self.label = self.slide_data.loc[:, 'train_mask'].dropna()
            self.label = self.label[:int(len(self.label)*self.dataset_cfg.label_ratio)]
            # self.preprocessor = A.Compose([
            #             # A.VerticalFlip(p=0.5),
            #             # A.RandomRotate90(p=0.5),
            #         ToTensorV2()])
            self.preprocessor = self._transform_(state)
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val_image'].dropna()
            self.data = self.data[:int(len(self.data)*self.dataset_cfg.label_ratio)]
            self.label = self.slide_data.loc[:, 'val_mask'].dropna()
            self.label = self.label[:int(len(self.label)*self.dataset_cfg.label_ratio)]
            # self.preprocessor = A.Compose([
            #             # A.VerticalFlip(p=0.5),
            #             # A.RandomRotate90(p=0.5),
            #         ToTensorV2()])
            self.preprocessor = self._transform_(state)
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test_image'].dropna()
            self.label = self.slide_data.loc[:, 'test_mask'].dropna()
            # self.preprocessor = A.Compose([A.Resize(height=self.image_size,width=self.image_size), 
            #                                             ToTensorV2()]) #暂时简单处理一下
            self.preprocessor = self._transform_(state)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        # image = self.preprocessor(image=image)["image"]
        # image = (image/127.5 - 1.0).to(torch.float32)
        # image = image.to(torch.float32)
        return image


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = dict()
        image_path = self.data[idx]
        # tensor_image = self.preprocess_image(image_path)
        tissue_type = str(Path(image_path).stem)

        label_path = self.label[idx]
        # label = np.load(label_path).astype('uint8')
        transformerd_data = self.preprocessor({'img': image_path, 'label':label_path})
        tensor_image = transformerd_data['img']
        tensor_label = transformerd_data['label'].squeeze()

        # # label = cv2.resize(
        # #     label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        # # )
        # # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
        # # If own dataset is used, then the below may need to be modified
        # tensor_label[(tensor_label == 3) | (tensor_label == 4)] = 3
        # tensor_label[(tensor_label == 5) | (tensor_label == 6) | (tensor_label == 7)] = 4

        # tensor_label = torch.from_numpy(label)

        example['data'] = tensor_image
        example['label'] = tensor_label
        example['image_path'] = image_path
        example['label_path'] = label_path
        example['tissue_type'] = tissue_type
        return example

    def load_img(self,image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image)
        return image.astype('float32')
    
    def load_ann(self,label_path):
        mask = np.load(label_path).astype('float32')
        if len(mask.shape) == 3: # pannuke
            num_class = 6
            label = mask
            mask_truth_buff = np.zeros_like(label)
            for class_idx in range(num_class):
                class_pred = label[class_idx]

                if class_idx != num_class-1: #last is the background
                    class_pred = np.clip(class_pred*(class_idx+1),0, (class_idx+1))
                else:
                    class_pred = np.clip(class_pred*0,0, 0)
                mask_truth_buff = mask_truth_buff + class_pred
            mask_truth = mask_truth_buff
            mask_truth[mask_truth>num_class-1]=0
            mask = np.max(mask_truth, axis=0)
        return mask

    def normalize(self,image):
        image = (image/127.5 - 1.0).to(torch.float32)
        return image

    def _transform_(self, state):
        # %% define transforms for image and segmentation
        if state == 'train':
            train_transforms = Compose(
                [
                    Lambdad(('img',), self.load_img),
                    Lambdad(('label',), self.load_ann),
                    # LoadImaged(
                    #     keys=["img", "label"], reader=PILReader, dtype=np.float32
                    # ),  # image three channels (H, W, 3); label: (H, W)
                    AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
                    AsChannelFirstd(
                        keys=["img"], channel_dim=-1, allow_missing_keys=True
                    ),  # image: (3, H, W)
                    # ScaleIntensityd(
                    # keys=["img"], allow_missing_keys=True
                    # ),  # Do not scale label
                    # SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
                    # RandSpatialCropd(
                    #     keys=["img", "label"], roi_size=args.input_size, random_size=False
                    # ),
                    RandAxisFlipd(keys=["img", "label"], prob=0.5),
                    RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
                    # # intensity transform
                    RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
                    RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
                    RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
                    RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
                    RandZoomd(
                        keys=["img", "label"],
                        prob=0.15,
                        min_zoom=0.5,
                        max_zoom=2.0,
                        mode=["area", "nearest"],
                    ),
                    Lambdad(('img',), self.normalize),
                    EnsureTyped(keys=["img", "label"]),
                ]
            )

            # transformed_data = apply_transform(train_transforms, {'img': load_img, 'label': load_ann})
            return train_transforms
        else:
            val_transforms = Compose(
                [
                    Lambdad(('img',), self.load_img),
                    Lambdad(('label',), self.load_ann),
                    # LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.float32),
                    AddChanneld(keys=["label"], allow_missing_keys=True),
                    AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
                    # ScaleIntensityd(keys=["img"], allow_missing_keys=True),
                    # AsDiscreted(keys=['label'], to_onehot=3),
                    # CenterSpatialCropd(
                    #     keys=["img", "label"], roi_size=args.input_size
                    # ),
                    Lambdad(('img',), self.normalize),
                    EnsureTyped(keys=["img", "label"]),
                ]
            )

            # transformed_data = apply_transform(val_transforms, {'img': load_img, 'label': load_ann})
            return val_transforms

