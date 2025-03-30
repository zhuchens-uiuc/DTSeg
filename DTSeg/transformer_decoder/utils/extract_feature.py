# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:52:16 2019

@author: ZeyuGao
"""
#提取所有图像的特征，用来给一步的方法做对比
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from torchvision import models
from PIL import ImageFile, Image
import pandas as pd
import glob 
import timm
import pickle
import random
from pathlib import Path
import torch.nn.functional as F
from collections import OrderedDict
import timm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ResNetNoGAP(nn.Module):
    def __init__(self, arch, num_classes=1000, zero_init_residual=False,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetNoGAP, self).__init__()
        # 创建原始的 ResNet 模型，但不包含分类器
        backbone = timm.create_model(arch, pretrained=False, num_classes=0,
                                     replace_stride_with_dilation=replace_stride_with_dilation,
                                     norm_layer=norm_layer)

        # 获取卷积后的特征图大小
        with torch.no_grad():
            fake_input = torch.randn(1, 3, 224, 224)
            feature_size = backbone.forward_features(fake_input).shape[-1]

        # # 创建新的分类器，用于代替原始模型中的全局平均池化层和 FC 层
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),  # 去掉原始模型中的全局平均池化层
        #     nn.Flatten(),
        #     nn.Linear(feature_size, num_classes)
        # )

        # 将新的分类器添加到模型最后面
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = self.classifier(x)
        return x




#---->遍历所有的WSI
class Dataset_All_Bags(data.Dataset):
    def __init__(self, wsi_path):
        self.image_all = sorted(glob.glob(wsi_path + '*'))
        self.slide = [Path(image).stem.split('_')[0] for image in self.image_all]
        self.slide_unique = np.unique(self.slide)
    
    def __len__(self):
        return len(self.slide_unique)

    def __getitem__(self, idx):
        select_slide = self.slide_unique[idx]
        slide_idx = np.where(np.array(self.slide)==select_slide)[0].tolist()
        return [self.image_all[s_idx] for s_idx in slide_idx]

#---->遍历所有的WSI
class Whole_Slide_Bag(data.Dataset):
    def __init__(self, wsi_path, transform=None):
        self.patch = wsi_path
        self.transform = transform
    
    def __len__(self):
        return len(self.patch)

    def __getitem__(self, idx):

        img = Image.open(self.patch[idx]).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img) 

        #---->坐标信息
        try:
            _, x, _, y = Path(self.patch[idx]).stem.split('/')[-1].split('_')[-4:]
            coord = [int(x), int(y)]
        except:
            coord = [0, 0]

        return img, coord, self.patch[idx]

def collate_features(batch):
    img = torch.stack([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    patch_dir = np.stack([item[2] for item in batch])
    return [img, coords, patch_dir]

# CUDA_VISIBLE_DEVICES=7 python extract_feature.py --wsi_files=/data114_2/shaozc/CellHE/MoNuSAC/Train_split/ --output_files='/data114_2/shaozc/CellHE/MoNuSAC/pt_files/Train_split/'
# CUDA_VISIBLE_DEVICES=6 python extract_feature.py --wsi_files=/data114_2/shaozc/CellHE/MoNuSAC/Test_split/ --output_files='/data114_2/shaozc/CellHE/MoNuSAC/pt_files/Test_split/'



ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser(description='RCC prediction')
parser.add_argument('--wsi_files', type=str, default='/data114_2/shaozc/CellHE/MoNuSAC/Train_split/')
parser.add_argument('--output_files', default='/data114_2/shaozc/CellHE/MoNuSAC/pt_files/Train_split/', type=str)
parser.add_argument('--file_path_base', type=str, help='model file path', default='/data114_2/shaozc/simsiam/checkpoints/resnet34/big/checkpoint_199.pth.tar')#result/resnet18-5c106cde.pth///data112/shaozc/checkpoint/camelyon_checkpoint_027.pth.tar
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--feature_size', default=512, type=int, help='')
# Miscs
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.gpu:
    torch.cuda.manual_seed_all(args.manualSeed)

if __name__ == '__main__':

    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(normMean, normStd)
    test_transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            normTransform
        ])

    # #---->加载模型
    # base_model = timm.create_model('vit_base_patch16_224_in21k', num_classes=0)
    # base_model = base_model.cuda()

    # checkpoint = torch.load(args.file_path_base)
    # # 加载21k权重时
    # del checkpoint['head.bias'], checkpoint['head.weight']
    # state_dict = checkpoint
    # # state_dict = checkpoint['model']
    # model_dict = base_model.state_dict()
    # weights = {k: v for k, v in state_dict.items() if k in model_dict}
    # model_dict.update(weights)
    # base_model.load_state_dict(model_dict)

    # checkpoint = torch.load(args.file_path_base)
    # state_dict = checkpoint['state_dict']
    # model_dict = base_model.state_dict()
    # weights = {k: v for k, v in state_dict.items() if k in model_dict}
    # model_dict.update(weights)
    # base_model.load_state_dict(model_dict)

    # state_dict = checkpoint['classifier_state_dict']
    # model_dict = classifier.state_dict()
    # weights = {k[7:]: v for k, v in state_dict.items() if k[7:] in model_dict}
    # # weights = {k: v for k, v in state_dict.items() if k in model_dict}
    # model_dict.update(weights)
    # classifier.load_state_dict(model_dict)

    # ResNet模型加载

    # base_model = timm.create_model('resnet34', pretrained=False, num_classes=0) #如果是预加载权重，pretrained=True
    base_model = ResNetNoGAP(arch='resnet34')

    checkpoint = torch.load(args.file_path_base)
    state_dict = checkpoint['state_dict']
    model_dict = base_model.state_dict()
    weights = {k[len('module.'):]: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(weights)
    msg = base_model.load_state_dict(model_dict)
    print(msg)

    base_model = base_model.to(device)
    # import timm
    # base_model = timm.create_model('regnetx_004', pretrained=True, num_classes=0) #如果是预加载权重，pretrained=True
    # base_model = base_model.to(device)

    #ResNet50模型加载，384维度
    # base_model = resnet50_baseline(pretrained=True)
    # base_model = base_model.to(device)


    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    base_model.eval()
    # classifier.eval()

    #---->遍历所有bag
    bags_dataset = Dataset_All_Bags(args.wsi_files)
    total = len(bags_dataset)


    #---->设置一下保存地址
    save_dir = Path(args.output_files)
    save_dir.mkdir(exist_ok=True, parents=True)
    dest_files = glob.glob(f'{save_dir}/*')
    # dest_files = [dest[:-5]+'.pt' for dest in dest_files]


    for bag_candidate_idx in range(total):
        slide_patches = bags_dataset[bag_candidate_idx]
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

        slide_id = Path(slide_patches[0]).stem.split('_')[0]
        if f'{save_dir}/{slide_id}.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 


        test_data = Whole_Slide_Bag(slide_patches, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=32, shuffle=False,
                num_workers=8, pin_memory=True, collate_fn=collate_features)


        with torch.no_grad():
            for batch_idx, (inputs, coord, patch_dir) in enumerate(tqdm(test_loader)):
                inputs = inputs.to(device, non_blocking=True)
                features = base_model(inputs)
                # logits = classifier(features)
                # output = F.softmax(logits, dim=1)
                # pseudo_label = torch.max(output, dim=1)[1]
                # #---->cpu
                features = features.cpu()
                # coord = torch.from_numpy(coord)
                # # #---->把所有的拼在一起
                batch_feature = features


                #---->保存一下
                patch_dir = patch_dir.tolist()
                for i, p_dir in enumerate(patch_dir):
                    p_save_name = Path(p_dir).stem 
                    torch.save(batch_feature[i, :].unsqueeze(0), f'{save_dir}/{p_save_name}.pt')
            # #---->concat
            # wsi_feature = torch.cat(wsi_feature)
            # slide_name = Path(slide_id).name
            # torch.save(wsi_feature, f'{save_dir}/{slide_name}.pt')