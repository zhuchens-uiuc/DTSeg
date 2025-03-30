from pathlib import Path

#---->read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers
def load_loggers(cfg):

    log_path = cfg.General.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = Path(cfg.config).parent 
    version_name = Path(cfg.config).name[:-5]
    cfg.log_path = Path(log_path) / log_name / version_name
    print(f'---->Log dir: {cfg.log_path}')
    
    # #---->TensorBoard
    # tb_logger = pl_loggers.TensorBoardLogger(log_path+str(log_name),
    #                                          name = version_name, version = f'fold{cfg.Data.fold}',
    #                                          log_graph = True, default_hp_metric = False)
    # return tb_logger
    # #---->CSV
    # csv_logger = pl_loggers.CSVLogger(log_path+str(log_name),
    #                                   name = version_name, version = f'fold{cfg.Data.fold}', )
    
    # return [tb_logger, csv_logger]



#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.distributed import rank_zero_only
import os
import glob

class MyModelCheckpoint(ModelCheckpoint):

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # Save latest
        if self.verbose > 0:
            epoch = trainer.current_epoch
            metrics = trainer.callback_metrics[self.monitor]
            latest_path = os.path.join(self.dirpath, 'last.ckpt')
            print(f"\nEpoch: {epoch}: Saving latest model to {latest_path}")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        # self._save_model(latest_path, trainer.model.model)
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.model.state_dict(),
                }, latest_path)
        except:
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.module.module.model.state_dict(),
                }, latest_path)
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if self.best_model_score==None or torch.gt(current, self.best_model_score):
            self.best_model_score = current
            bestloss_path = os.path.join(self.dirpath, 'epoch={:02d}-val_IoU={:.4f}.ckpt'.format(trainer.current_epoch, trainer.callback_metrics[self.monitor]))
            if self.verbose > 0:
                print(f"Saving best model to {bestloss_path}")
            
            exist_bast_path = glob.glob(self.dirpath+'/epoch*')
            for exist_path in exist_bast_path:
                os.remove(exist_path)
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.model.state_dict(),
                    }, bestloss_path)
            except:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.module.module.model.state_dict(),
                    }, bestloss_path)

            try: #ema
                bestloss_ema_path = os.path.join(self.dirpath, 'epoch={:02d}-val_IoU={:.4f}-ema.ckpt'.format(trainer.current_epoch, trainer.callback_metrics[self.monitor]))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.model_ema.state_dict(),
                    }, bestloss_ema_path)
            except:
                pass


class MyModelCheckpointTwo(ModelCheckpoint):

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # Save latest
        if self.verbose > 0:
            epoch = trainer.current_epoch
            metrics = trainer.callback_metrics[self.monitor]
            latest_path = os.path.join(self.dirpath, 'last.ckpt')
            latest_image_path = os.path.join(self.dirpath, 'last_image.ckpt')
            print(f"\nEpoch: {epoch}: Saving latest model to {latest_path}")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        # self._save_model(latest_path, trainer.model.model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.model.state_dict(),
            }, latest_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.model_image.state_dict(),
            }, latest_image_path)

        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if self.best_model_score==None or torch.gt(current, self.best_model_score):
            self.best_model_score = current
            bestloss_path = os.path.join(self.dirpath, 'epoch={:02d}-val_IoU={:.4f}.ckpt'.format(trainer.current_epoch, trainer.callback_metrics[self.monitor]))
            bestloss_image_path = os.path.join(self.dirpath, 'epoch={:02d}-val_IoU={:.4f}_Image.ckpt'.format(trainer.current_epoch, trainer.callback_metrics[self.monitor]))
            if self.verbose > 0:
                print(f"Saving best model to {bestloss_path}")
            
            exist_bast_path = glob.glob(self.dirpath+'/epoch*')
            for exist_path in exist_bast_path:
                os.remove(exist_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.model.state_dict(),
                }, bestloss_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.model_image.state_dict(),
                }, bestloss_image_path)


class MyModelCheckpointUnlabel(ModelCheckpoint):

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.train_dataset.set_unlabel_data(trainer.current_epoch)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # Save latest
        if self.verbose > 0:
            epoch = trainer.current_epoch
            metrics = trainer.callback_metrics[self.monitor]
            latest_path = os.path.join(self.dirpath, 'last.ckpt')
            print(f"\nEpoch: {epoch}: Saving latest model to {latest_path}")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        # self._save_model(latest_path, trainer.model.model)
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.model.state_dict(),
                }, latest_path)
        except:
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.module.module.model.state_dict(),
                }, latest_path)
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if self.best_model_score==None or torch.gt(current, self.best_model_score):
            self.best_model_score = current
            bestloss_path = os.path.join(self.dirpath, 'epoch={:02d}-val_IoU={:.4f}.ckpt'.format(trainer.current_epoch, trainer.callback_metrics[self.monitor]))
            if self.verbose > 0:
                print(f"Saving best model to {bestloss_path}")
            
            exist_bast_path = glob.glob(self.dirpath+'/epoch*')
            for exist_path in exist_bast_path:
                os.remove(exist_path)
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.model.state_dict(),
                    }, bestloss_path)
            except:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.module.module.model.state_dict(),
                    }, bestloss_path)

            try: #ema
                bestloss_ema_path = os.path.join(self.dirpath, 'epoch={:02d}-val_IoU={:.4f}-ema.ckpt'.format(trainer.current_epoch, trainer.callback_metrics[self.monitor]))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.model_ema.state_dict(),
                    }, bestloss_ema_path)
            except:
                pass



def load_callbacks(cfg):

    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_IoU',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='max'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(MyModelCheckpoint(monitor = 'val_IoU',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_IoU:.4f}',
                                         verbose = True,
                                         save_last = False,
                                         save_top_k = 0,
                                         mode = 'max',
                                         save_weights_only = True))
        
    return Mycallbacks

def load_callbacks_two(cfg):

    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_IoU',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='max'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(MyModelCheckpointTwo(monitor = 'val_IoU',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_IoU:.4f}',
                                         verbose = True,
                                         save_last = False,
                                         save_top_k = 0,
                                         mode = 'max',
                                         save_weights_only = True))
        
    return Mycallbacks

def load_callbacks_unlabel(cfg):

    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_IoU',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='max'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(MyModelCheckpointUnlabel(monitor = 'val_IoU',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_IoU:.4f}',
                                         verbose = True,
                                         save_last = False,
                                         save_top_k = 0,
                                         mode = 'max',
                                         save_weights_only = True))
        
    return Mycallbacks

#---->val loss
import torch
import torch.nn.functional as F
def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i]) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    loss = - torch.sum(x_log) / len(y)
    return loss

#---->mean Iou
#https://cvnote.ddlee.cc/2020/07/22/image-segmentation-evaluation
import numpy as np

def intersect_and_union(pred_label, label, num_classes, ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index=ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union
    mean_Iou = np.mean(iou)

    return mean_Iou


#----> visual the prediction
from PIL import Image
palette = [ 
  255,  255, 255, # bg
  59,103,188, # bed
  234,113,43,     # bed footboard
  155,155,155,   # bed headboard
  255,184,2,  # bed side rail
  80,144,207,  # carpet
  101,163,62,  # ceiling
  101,43,150,  # chandelier / ceiling fan blade
  0,0,0,  # curtain
  99 , 83  , 3,    # cushion
  116 , 116 , 138, # floor
  63  ,182 , 24,   # table/nightstand/dresser
  200  ,226 , 37,  # table/nightstand/dresser top
  225 , 184 , 161, # picture / mirrow
  233 ,  5  ,219,  # pillow
  142 , 172  ,248, # lamp column
  153 , 112 , 146, # lamp shade
  38  ,112 , 254,  # wall
  229 , 30  ,141,  # window
  99, 205, 255,    # curtain rod
  74, 59, 83,      # window frame
  186, 9, 0,       # chair
  107, 121, 0,     # picture / mirrow frame
  0, 194, 160,     # plinth
  255, 170, 146,   # door / door frame
  255, 144, 201,   # pouf
  185, 3, 170,     # wardrobe
  221, 239, 255,   # plant
  0, 0, 53,        # table staff
]

def get_palette(num_class):
    # select_idx = np.random.choice(range(len(palette)), num_class, replace=False)
    select_idx = np.arange(num_class*3)
    return [palette[idx] for idx in select_idx]

def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


#######visual 
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


######concat all the predict images
####
def _post_process_patches(
    post_proc_func, post_proc_kwargs, patch_info, image_info, overlay_kwargs,
):
    """Apply post processing to patches.
    
    Args:
        post_proc_func: post processing function to use
        post_proc_kwargs: keyword arguments used in post processing function
        patch_info: patch data and associated information
        image_info: input image data and associated information
        overlay_kwargs: overlay keyword arguments

    """
    # re-assemble the prediction, sort according to the patch location within the original image
    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
    patch_info, patch_data = zip(*patch_info)

    src_shape = image_info["src_shape"]
    src_image = image_info["src_image"]

    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_info]) + 1
    nr_col = max([x[3] for x in patch_info]) + 1
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(
        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
    )
    # crop back to original shape
    pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])

    # * Implicit protocol
    # * a prediction map with instance of ID 1-N
    # * and a dict contain the instance info, access via its ID
    # * each instance may have type
    pred_inst, inst_info_dict = post_proc_func(pred_map, **post_proc_kwargs)

    overlaid_img = visualize_instances_dict(
        src_image.copy(), inst_info_dict, **overlay_kwargs
    )

    return image_info["name"], pred_map, pred_inst, inst_info_dict, overlaid_img