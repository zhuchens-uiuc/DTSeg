
import os
import glob
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class CelebAHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        # root = "/data114_2/shaozc/LiveCell/images/livecell_train_val_images/"
        # paths = sorted(glob.glob(root + '*.tif'))
        # root = "/data114_2/shaozc/CellHE/Consep/CoNSeP/Train_split_aug2/"
        # paths = sorted(glob.glob(root + '*.png'))
        root = "/data114_3/shaozc/Dataset/BCI_dataset/IHC/train_split_256_128/"
        paths = sorted(glob.glob(root + '*.png'))
        paths = paths[:int(0.9*len(paths))]

        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        # root = "/data114_2/shaozc/LiveCell/images/livecell_train_val_images/"
        # paths = sorted(glob.glob(root + '*.tif'))
        # root = "/data114_2/shaozc/CellHE/Consep/CoNSeP/Train_split_aug2/"
        # paths = sorted(glob.glob(root + '*.png'))
        root = "/data114_3/shaozc/Dataset/BCI_dataset/IHC/train_split_256_128/"
        paths = sorted(glob.glob(root + '*.png'))
        paths = paths[int(0.9*len(paths)):]

        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys