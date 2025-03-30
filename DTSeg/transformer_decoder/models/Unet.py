import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(self, num_class):
        super(Unet, self).__init__()
        self.model = smp.Unet('resnet34', classes=num_class)


    def forward(self, x):
        x = self.model(x)
        results_dict = {'logits': x, 'Y_probs': F.softmax(x, dim = 1), 'Y_hat': torch.topk(x, 1, dim = 1)[1]}
        return results_dict