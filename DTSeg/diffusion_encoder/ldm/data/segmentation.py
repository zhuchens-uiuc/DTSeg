
import os
import glob
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset

class ImagePaths(Dataset):
    def __init__(self, paths, mask_paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self.labels["file_mask_path_"] = mask_paths
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
    
    def preprocess_mask(self,label_path):
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

    def __getitem__(self, i):
        example = dict()
        example["mask"] = self.preprocess_image(self.labels["file_path_"][i])
        example["image"] = self.preprocess_mask(self.labels["file_mask_path_"][i])

        example["image"][example["image"]>0]=1

        example["image"] = np.stack((example["image"], example["image"], example["image"]), axis=-1)

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
        root = "/data114_2/shaozc/CellHE/PanNuke/Fold2/images/fold2/images/"
        paths = sorted(glob.glob(root + '*.png'))
        mask_paths = [path.replace("images", "masks").replace('png', 'npy') for path in paths]

        self.data = ImagePaths(paths=paths, mask_paths=mask_paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        # root = "/data114_2/shaozc/LiveCell/images/livecell_train_val_images/"
        # paths = sorted(glob.glob(root + '*.tif'))
        # root = "/data114_2/shaozc/CellHE/Consep/CoNSeP/Train_split_aug2/"
        # paths = sorted(glob.glob(root + '*.png'))
        root = "/data114_2/shaozc/CellHE/PanNuke/Fold3/images/fold3/images/"
        paths = sorted(glob.glob(root + '*.png'))
        mask_paths = [path.replace("images", "masks").replace('png', 'npy') for path in paths]

        self.data = ImagePaths(paths=paths, mask_paths=mask_paths, size=size, random_crop=False)
        self.keys = keys