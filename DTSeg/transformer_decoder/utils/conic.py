import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import cv2
import sys
from tqdm import tqdm
from os.path import join as opj

def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).

    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.
        
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

data_root = "/data114_2/shaozc/CellHE/CoNIC" #! Change this according to the root path where the data is located

images_path = "%s/images.npy" % data_root # images array Nx256x256x3
labels_path = "%s/labels.npy" % data_root # labels array Nx256x256x3
counts_path = "%s/counts.csv" % data_root # csv of counts per nuclear type for each patch
info_path = "%s/patch_info.csv" % data_root # csv indicating which image from Lizard each patch comes from


images = np.load(images_path)
labels = np.load(labels_path)
counts = pd.read_csv(counts_path)
patch_info = pd.read_csv(info_path)

print("Images Shape:", images.shape)
print("Labels Shape:", labels.shape)

dataset_info = patch_info['patch_info'].map(lambda x: x.split('_')[0]).tolist()
consep_idx = np.where(np.array(dataset_info) == 'consep')[0].tolist()
pannuke_idx = np.where(np.array(dataset_info) == 'pannuke')[0].tolist()

count_cell = 0
for idx in tqdm(range(len(images))):

    if idx in consep_idx or idx in pannuke_idx:
        continue

    patch_img = images[idx] # 256x256x3
    patch_lab = labels[idx] # 256x256x2
    patch_inst_map = patch_lab[..., 0]
    patch_class_map = patch_lab[..., 1]
    patch_name = patch_info['patch_info'][idx]
    count_cell = count_cell + np.max(patch_inst_map)
    # cv2.imwrite(opj('/data114_2/shaozc/CellHE/CoNIC/', 'images', f'{patch_name}.png'), patch_img)
    # np.save(opj('/data114_2/shaozc/CellHE/CoNIC/', 'labels', f'{patch_name}.npy'), patch_class_map)
print(count_cell)