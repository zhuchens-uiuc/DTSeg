



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange
from timm.models.vision_transformer import Mlp
from collections import OrderedDict



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class AggregrateWSA(nn.Module):
    def __init__(self, num_class, feature_channels, image_channels, decode_channels, dropout=0.1, num_heads=8):
        super(AggregrateWSA, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(feature_channels, decode_channels), nn.ReLU()) 
        self.fc2 = nn.Sequential(nn.Linear(image_channels, decode_channels), nn.ReLU()) 
        # # self.eca_layer = eca_layer()
        # self.multihead_attn = nn.MultiheadAttention(decode_channels, num_heads)
        # self.MLP = Mlp(in_features=decode_channels, hidden_features=decode_channels*4, act_layer=nn.GELU, drop=0.1)
        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels*2, decode_channels),
                                        nn.Dropout2d(p=dropout, inplace=True),
                                        )
        self.segmentation = nn.Sequential(OrderedDict([('2', Conv(decode_channels, num_class, kernel_size=1))]))
        
        # self.auxiliary_head = nn.Sequential(#ConvBNReLU(decode_channels*2, decode_channels),
        #                        #nn.Dropout2d(p=dropout, inplace=True),
        #                         Conv(decode_channels, 2, kernel_size=1),
        #                         nn.AdaptiveAvgPool2d(1))


    def forward(self, x_1, x_2, return_unlabel=False):
        #F.interpolate(mask, (800, 1200), mode='nearest')
        if x_1.shape[-1] != 256:
            x_1 = F.interpolate(x_1, (256, 256), mode='nearest')
        if x_2.shape[-1] != 256:
            x_2 = F.interpolate(x_2, (256, 256), mode='nearest')

        x_1 = self.fc1(x_1.flatten(2).transpose(1, 2))
        x_2 = self.fc2(x_2.flatten(2).transpose(1, 2))
        # attn_output, attn_output_weights = self.multihead_attn(x_1, x_2, x_2)
        # attn_output = self.MLP(attn_output)
        # attn_output = attn_output.transpose(1, 2).view(attn_output.shape[0], -1, 256, 256)
        x_1 = x_1.transpose(1, 2).view(x_1.shape[0], -1, 256, 256)
        x_2 = x_2.transpose(1, 2).view(x_2.shape[0], -1, 256, 256)
        x = self.segmentation(self.segmentation_head(torch.cat((x_1, x_2), dim=1)))
        x_1_feature = self.segmentation_head(torch.cat((x_1, x_1), dim=1))
        x_2_feature = self.segmentation_head(torch.cat((x_2, x_2), dim=1))

        if return_unlabel == True:
            out1 = self.segmentation(self.segmentation_head(torch.cat((x_1, x_1), dim=1)))
            out2 = self.segmentation(self.segmentation_head(torch.cat((x_2, x_2), dim=1)))
            x_1_feature = self.segmentation_head(torch.cat((x_1, x_1), dim=1))
            x_2_feature = self.segmentation_head(torch.cat((x_2, x_2), dim=1))
            return out1, out2, x_1_feature, x_2_feature

        results_dict = {'logits': x, 'Y_probs': F.softmax(x, dim = 1), 'Y_hat': torch.topk(x, 1, dim = 1)[1], 'x_1_feature': x_1_feature, 'x_2_feature': x_2_feature}
        return results_dict

        # batch_size = x_1.shape[0]
        # auxiliary = 0
        # for i in range(batch_size):
        #     random_idx = np.random.choice(np.array((0,1)),1)
        #     if np.sum(random_idx) == 0:
        #         random_label=0
        #         # x_auxiliary = self.auxiliary_head(torch.cat((x_1[i].unsqueeze(0), x_1[i].unsqueeze(0)), dim=1))
        #         x_auxiliary = self.auxiliary_head(self.segmentation_head(torch.cat((x_1[i].unsqueeze(0), x_1[i].unsqueeze(0)), dim=1)))
        #     elif np.sum(random_idx) == 1:
        #         random_label=1
        #         # x_auxiliary = self.auxiliary_head(torch.cat((x_2[i].unsqueeze(0), x_2[i].unsqueeze(0)), dim=1))
        #         x_auxiliary = self.auxiliary_head(self.segmentation_head(torch.cat((x_2[i].unsqueeze(0), x_2[i].unsqueeze(0)), dim=1)))

        #     auxiliary = auxiliary + F.cross_entropy(x_auxiliary.squeeze().unsqueeze(0), torch.tensor([random_label]).cuda())
        # # x = self.segmentation_head(torch.cat((x_1, x_2), dim=1))
        # results_dict = {'logits': x, 'Y_probs': F.softmax(x, dim = 1), 'Y_hat': torch.topk(x, 1, dim = 1)[1], 'auxiliary': auxiliary/batch_size}
        # return results_dict



# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from PIL import Image
# from einops import rearrange
# from timm.models.vision_transformer import Mlp



# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels),
#             nn.ReLU6()
#         )

# class Conv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
#         super(Conv, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
#         )

# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)

#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

#         # Multi-scale information fusion
#         y = self.sigmoid(y)

#         return x * y.expand_as(x)


# class AggregrateWSA(nn.Module):
#     def __init__(self, num_class, feature_channels, image_channels, decode_channels, dropout=0.1, num_heads=8):
#         super(AggregrateWSA, self).__init__()
#         self.fc1 = nn.Sequential(nn.Linear(feature_channels, decode_channels), nn.ReLU()) 
#         self.fc2 = nn.Sequential(nn.Linear(image_channels, decode_channels), nn.ReLU()) 
#         # # self.eca_layer = eca_layer()
#         # self.multihead_attn = nn.MultiheadAttention(decode_channels, num_heads)
#         # self.MLP = Mlp(in_features=decode_channels, hidden_features=decode_channels*4, act_layer=nn.GELU, drop=0.1)
#         self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels*2, decode_channels),
#                                         nn.Dropout2d(p=dropout, inplace=True),
#                                         Conv(decode_channels, num_class, kernel_size=1))


#     def forward(self, x_1, x_2):
#         #F.interpolate(mask, (800, 1200), mode='nearest')
#         if x_1.shape[-1] != 256:
#             x_1 = F.interpolate(x_1, (256, 256), mode='nearest')
#         if x_2.shape[-1] != 256:
#             x_2 = F.interpolate(x_2, (256, 256), mode='nearest')

#         x_1 = self.fc1(x_1.flatten(2).transpose(1, 2))
#         x_2 = self.fc2(x_2.flatten(2).transpose(1, 2))
#         # attn_output, attn_output_weights = self.multihead_attn(x_1, x_2, x_2)
#         # attn_output = self.MLP(attn_output)
#         # attn_output = attn_output.transpose(1, 2).view(attn_output.shape[0], -1, 256, 256)
#         x_1 = x_1.transpose(1, 2).view(x_1.shape[0], -1, 256, 256)
#         x_2 = x_2.transpose(1, 2).view(x_2.shape[0], -1, 256, 256)
#         x = self.segmentation_head(torch.cat((x_1, x_2), dim=1))
#         # x = self.segmentation_head(torch.cat((x_1, x_2), dim=1))
#         results_dict = {'logits': x, 'Y_probs': F.softmax(x, dim = 1), 'Y_hat': torch.topk(x, 1, dim = 1)[1]}
#         return results_dict