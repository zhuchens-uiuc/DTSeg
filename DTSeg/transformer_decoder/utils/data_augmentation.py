import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import myTransforms
from pathlib import Path
from tqdm import tqdm


def img_trans(img, img_name, time=0, SAVE=False):
    # preprocess = myTransforms.Compose([
    #     myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=1),
    #                                myTransforms.RandomVerticalFlip(p=1),
    #                                myTransforms.AutoRandomRotation()]),  # above is for: randomly selecting one for process
    #     # myTransforms.RandomAffine(degrees=0, translate=[0, 0.2], scale=[0.8, 1.2],
    #     #                           shear=[-10, 10, -10, 10], fillcolor=(228, 218, 218)),
    #     myTransforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
    #     myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0, 2), hue=0.3),
    #                                myTransforms.HEDJitter(theta=0.05)]),
    #     # myTransforms.ToTensor(),  #operated on original image, rewrite on previous transform.
    #     # myTransforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
    # ])
    # print(preprocess)

    preprocess1 = myTransforms.HEDJitter(theta=random.sample(range(1, 10), 1)[0]/100)
    # print(preprocess1)
    preprocess2 = myTransforms.RandomGaussBlur(radius=[0.5, 2.5])
    # print(preprocess2)
    preprocess3 = myTransforms.RandomAffineCV2(alpha=random.sample(range(1, 15), 1)[0]/100)  # alpha \in [0,0.15]
    # print(preprocess3)
    preprocess4 = myTransforms.RandomElastic(alpha=random.sample(range(1, 5), 1)[0]/100, sigma=0.1)
    # print(preprocess4)
    preprocess5 = myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=1),
                                   myTransforms.RandomVerticalFlip(p=1),
                                   myTransforms.AutoRandomRotation()])  # above is for: randomly selecting one for process
    # print(preprocess5)
    preprocess6 = myTransforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5))
    # print(preprocess6)

    # composeimg = preprocess(img)
    HEDJitterimg = preprocess1(img)
    blurimg = preprocess2(img)
    affinecvimg = preprocess3(img)
    elasticimg = preprocess4(img,mask=None)
    Rotationimg = preprocess5(img)
    Colorimg = preprocess6(img)


    root_save = '/data114_2/shaozc/CellHE/Consep/CoNSeP/Train_split_aug/'
    # root_save = '/data114_2/shaozc/unitopath-public/512'
    Path(root_save).mkdir(exist_ok=True, parents=True)
    if SAVE:
        HEDJitterimg.save(root_save + f'/{img_name}_HEDJitter_{str(time)}_.png')
        blurimg.save(root_save + f'/{img_name}_blurimg_{str(time)}_.png')
        affinecvimg.save(root_save + f'/{img_name}_affinecvimg_{str(time)}_.png')
        elasticimg.save(root_save + f'/{img_name}_elasticimg_{str(time)}_.png')
        Rotationimg.save(root_save + f'/{img_name}_Rotationimg_{str(time)}_.png')
        Colorimg.save(root_save + f'/{img_name}_Colorimg_{str(time)}_.png')
    # else:
    #     plt.subplot(321)
    #     plt.imshow(img)
    #     plt.subplot(322)
    #     plt.imshow(composeimg)
    #     plt.subplot(323)
    #     plt.imshow(HEDJitterimg)
    #     plt.subplot(324)
    #     plt.imshow(blurimg)
    #     plt.subplot(325)
    #     plt.imshow(affinecvimg)
    #     plt.subplot(326)
    #     plt.imshow(elasticimg)
    #     plt.show()
    #     plt.close()

class BaseDataset(Dataset):

    def __init__(self, times):
        self.img_dir = '/data114_2/shaozc/CellHE/Consep/CoNSeP/Train_split/'
        self.data = list(Path(self.img_dir).glob('*.png')) #所有的原图
        self.times = times

        # self.img_dir = '/data114_2/shaozc/unitopath-public/512/images/'
        # self.data = list(Path(self.img_dir).glob('*.png')) #所有的原图
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample_i = self.data[i]
        img = Image.open(sample_i)
        img_trans(img, Path(sample_i).stem, time=self.times, SAVE=True)
        return 0

if __name__ == '__main__':
    for i in range(0, 20):

        train_dataset = BaseDataset(times=i)
        batch_size = 20
        train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    collate_fn=None, # 一般不用设置
                                    num_workers=batch_size)

        for step,_ in enumerate(tqdm(train_loader)):
            pass