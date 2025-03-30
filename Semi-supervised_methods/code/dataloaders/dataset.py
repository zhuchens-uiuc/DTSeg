import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from PIL import Image
from pathlib import Path



class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample
    
class BaseDataSetsPro(Dataset):
    def __init__(
        self,
        base_dir=None,
        csv_dir=None, 
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.csv_dir = csv_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            df = pd.read_csv(self._base_dir + "/" + csv_dir, index_col=0)
            self.sample_list = df.loc[:, 'train_image'].dropna()
            self.sample_mask_list = df.loc[:, 'train_mask'].dropna()

        elif self.split == "val":
            df = pd.read_csv(self._base_dir + "/" + csv_dir, index_col=0)
            self.sample_list = df.loc[:, 'val_image'].dropna()
            self.sample_mask_list = df.loc[:, 'val_mask'].dropna()
        elif self.split == "test":
            df = pd.read_csv(self._base_dir + "/" + csv_dir, index_col=0)
            self.sample_list = df.loc[:, 'test_image'].dropna()
            self.sample_mask_list = df.loc[:, 'test_mask'].dropna()
        if num is not None: # and self.split == "train":
            self.sample_list = self.sample_list[:int(len(self.sample_list)*num)]
            self.sample_mask_list = self.sample_mask_list[:int(len(self.sample_mask_list)*num)]
        print("total {} samples".format(len(self.sample_list)))

    def normalize(self,image):
        image = (image/127.5 - 1.0)
        return image
    def load_img(self,image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image)
        return (image.astype('float32'))
    
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
        return (mask)
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_path = self.sample_list[idx]
        label_path = self.sample_mask_list[idx]
        tensor_image = self.load_img(image_path)
        tensor_label = self.load_ann(label_path)
        tissue_type = str(Path(image_path).stem)

        sample = {"image": tensor_image, "label": tensor_label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        if self.split == 'test':
            sample = {"image": tensor_image, "label": tensor_label, "tissue_type": tissue_type, "image_path": image_path}

        #normalize
        try:
            sample['image'] = self.normalize(sample['image']).permute(2, 0, 1)
            if 'image_weak' in sample.keys(): 
                sample['image_weak'] = self.normalize(sample['image_weak']).permute(2, 0, 1)
        except:
            sample['image'] = torch.from_numpy(sample['image'].astype(np.float32))
            sample['label'] = torch.from_numpy(sample['label'].astype(np.uint8))
            sample['image'] = self.normalize(sample['image']).permute(2, 0, 1)
        sample["idx"] = idx

        if self.csv_dir == 'consep_split.csv' or self.csv_dir == 'consep_split_42.csv' or self.csv_dir == 'consep_split_107.csv':
            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            sample['label'][(sample['label'] == 3) | (sample['label'] == 4)] = 3
            sample['label'][(sample['label'] == 5) | (sample['label'] == 6) | (sample['label'] == 7)] = 4            

        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # x, y = image.shape
        # image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # image = self.resize(image)
        # label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32))
        image_weak = torch.from_numpy(image_weak.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
