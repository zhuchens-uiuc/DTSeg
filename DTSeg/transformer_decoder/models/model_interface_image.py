import os
import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import mean_iou, get_palette, colorize_mask
from utils.stats_utils import get_fast_aji_plus, get_fast_pq, get_dice_1
from utils.utils import get_bounding_box
from PIL import Image
from utils.hover_utils import get_inst_centroid
from models.HoVerNetML import compute_hv_map
from models.HoVerNetML import _post_process_single_hovernet, _convert_multiclass_mask_to_binary

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


#----> Metrics
import segmentation_models_pytorch as smp


#---->
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only

from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# #---->define the feature extractor
# steps = [50,150,250]
# blocks = [5,6,7,8,12]
# model_path = '/data114_2/shaozc/LiveCell/latent-diffusion-main/logs/2023-02-06T21-52-07_livecell256/checkpoints/epoch=000268.ckpt'
# model_config = '/data114_2/shaozc/LiveCell/latent-diffusion-main/configs/latent-diffusion/livecell256.yaml'
# input_activations = False
# feature_extractor = FeatureExtractorLDM(dict({'steps':steps, 'blocks':blocks, 'model_path':model_path, 'model_config':model_config, 'input_activations':input_activations}))


class ModelInterfaceImage(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterfaceImage, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.num_class = model.num_class
        self.log_path = kargs['log']
        self.dataset_cfg = kargs['data']
        
        #---->Metrics
        # metrics = torchmetrics.MetricCollection([
        #                                             torchmetrics.IoU(num_classes=self.num_class),
        #                                         ])
        # self.valid_metrics = metrics.clone(prefix = 'val_')
        # self.test_metrics = metrics.clone(prefix = 'test_')


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


    def training_step(self, batch, batch_idx):

        results_dict = self.model(batch['data'])
        logits = results_dict['logits']
        

        loss = self.loss(logits.float(), batch['label'].type(torch.int64))


        return {'loss': loss} 

    # def training_epoch_end(self, training_step_outputs):
    #     for c in range(self.num_class):
    #         count = self.data[c]["count"]
    #         correct = self.data[c]["correct"]
    #         if count == 0: 
    #             acc = None
    #         else:
    #             acc = float(correct) / count
    #         print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
    #     self.data = [{"count": 0, "correct": 0} for i in range(self.num_class)]


    def validation_step(self, batch, batch_idx):
        results_dict = self.model(batch['data'])
        logits = results_dict['logits'].detach().cpu()
        probs = results_dict['Y_probs'].detach().cpu()
        Y_hat = results_dict['Y_hat'].detach().cpu()
        return {'logits' : logits, 'Y_probs': probs, 'Y_hat': Y_hat, 'label' : batch['label']}


    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_probs'] for x in val_step_outputs], dim = 0)
        max_probs = torch.cat([x['Y_hat'] for x in val_step_outputs], dim = 0)
        target = torch.cat([x['label'] for x in val_step_outputs], dim = 0)
        
        #---->
        metrics = dict()
        # metrics['val_IoU'] = mean_iou(max_probs.squeeze().cpu().numpy() , target.cpu().numpy(), self.num_class, 0)
        tp, fp, fn, tn = smp.metrics.get_stats(max_probs.squeeze().cpu(), target.cpu().long(), mode='multiclass', num_classes=self.num_class)

        # then compute metrics with required reduction (see metric docs)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        # f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        # accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        # recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

        metrics['val_IoU'] = iou_score
        metrics['val_f1'] = f1_score
        # metrics['val_f2'] = f2_score
        # metrics['val_acc'] = accuracy
        # metrics['val_recall'] = recall

        self.log_dict(metrics, on_epoch = True, logger = True)

        


    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]


    def test_step(self, batch, batch_idx):

        results_dict = self.model(batch['data'])
        logits = results_dict['logits'].detach().cpu()
        probs = results_dict['Y_probs'].detach().cpu()
        Y_hat = results_dict['Y_hat'].detach().cpu()

        # if self.hparams.state == 'pseudo':
        #---->image log
        batch_size = self.dataset_cfg.test_dataloader.batch_size
        image_path = batch['image_path']

        os.makedirs(os.path.join(self.log_path, 'logits'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(self.log_path, 'visualizations'), exist_ok=True)

        self.palette = get_palette(self.num_class)

        for i, logit in enumerate(logits):
            filename = image_path[i].split('/')[-1].split('.')[0]
            np.save(os.path.join(self.log_path, 'logits', filename + '.npy'), logit.numpy())

        for i, pred in enumerate(Y_hat):
            filename = image_path[i].split('/')[-1].split('.')[0]
            pred = pred.view(self.dataset_cfg.image_size, self.dataset_cfg.image_size).numpy()
            np.save(os.path.join(self.log_path, 'predictions', filename + '.npy'), pred)

            mask = colorize_mask(pred, self.palette)
            Image.fromarray(mask).save(
                os.path.join(self.log_path, 'visualizations', filename + '.jpg')
            )


        return {'logits' : logits, 'Y_probs': probs, 'Y_hat': Y_hat, 'label' : batch['label'], 'tissue': batch['tissue_type']}

    def test_epoch_end(self, output_results):
        logits = torch.cat([x['logits'] for x in output_results], dim = 0)
        probs = torch.cat([x['Y_probs'] for x in output_results], dim = 0)
        max_probs = torch.cat([x['Y_hat'] for x in output_results], dim = 0)
        target = torch.cat([x['label'] for x in output_results], dim = 0)
        tissue_type = np.concatenate([x['tissue'] for x in output_results])
        
        #---->
        metrics = dict()
        # metrics['Mean_IoU'] = mean_iou(max_probs.squeeze().cpu().numpy() , target.cpu().numpy(), self.num_class, 0)

        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(max_probs.squeeze().cpu(), target.cpu().long(), mode='multiclass', num_classes=self.num_class)

        # then compute metrics with required reduction (see metric docs)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        # f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        # accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        # recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        # macro_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        # macro_f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")

        # DICE = get_dice_1(target.cpu().numpy(), max_probs.squeeze().cpu().numpy())
        # AJI = get_fast_aji_plus(target.cpu().numpy(), max_probs.squeeze().cpu().numpy())
        # # DQ, SQ, PQ = get_fast_pq(target.cpu().numpy(), max_probs.squeeze().cpu().numpy())[0]
        # metrics['DICE'] = DICE
        # metrics['AJI'] = AJI
        # metrics['DQ'] = DQ
        # metrics['SQ'] = SQ
        # metrics['PQ'] = PQ


        if self.hparams.state != 'pseudo':

            for ins_idx in tqdm(range(len(max_probs))):

                mask = max_probs[ins_idx, 0].numpy()

                save_parent = self.log_path / 'pred'
                save_parent.mkdir(exist_ok=True)
                mask = mask.astype(np.uint8) #(256, 256)


                # # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
                # # If own dataset is used, then the below may need to be modified
                # mask[(mask == 3) | (mask == 4)] = 3
                # mask[(mask == 5) | (mask == 6) | (mask == 7)] = 4

                mask_buff = np.zeros((self.num_class,mask.shape[0], mask.shape[1]), dtype=np.uint8)
                nucleus_labels = list(np.unique(mask))
                if 0 in nucleus_labels:
                    nucleus_labels.remove(0)  # 0 is background

                # for each nucleus, get the class predictions for the pixels and take a vote
                for nucleus_ix in nucleus_labels:
                    # get mask for the specific nucleus
                    ix_mask = mask == nucleus_ix
                    votes = mask[ix_mask]
                    majority_class = np.argmax(np.bincount(votes))
                    mask_buff[majority_class-1][ix_mask] = nucleus_ix
                mask_buff[-1][mask==0]=1
                # mask = mask_buff

                mask_1c = np.sum(mask_buff[:-1], axis=0)
                hv_map = compute_hv_map(mask_1c)
                hv_map = torch.from_numpy(hv_map)
            # if self.state == 'test':
                small_obj_size_thresh = 10
                kernel_size = 21
                h=0.5 
                k=0.5
                mask_1c[mask_1c>0]=1
                true_mask = np.stack((1-mask_1c, mask_1c), axis=0)
                true_mask = np.array(true_mask, dtype=np.float64)
                true_mask = torch.from_numpy(true_mask)
                instance_pred = _post_process_single_hovernet(
                            true_mask, hv_map, small_obj_size_thresh, kernel_size, h, k
                        )

                # mask_truth_buff = np.zeros_like(mask)
                # for class_idx in range(args.num_class):
                #     class_pred = mask[class_idx]

                #     if class_idx != args.num_class-1: #last is the background
                #         class_pred = np.clip(class_pred*(class_idx+1),0, (class_idx+1))
                #     else:
                #         class_pred = np.clip(class_pred*0,0, 0)
                #     mask_truth_buff[class_idx] = class_pred
                # mask_pred = mask_truth_buff
                # mask_pred[mask_pred>args.num_class]=0
                # mask_pred = np.argmax(mask, axis=0)
                mask_pred = mask


                inst_info_dict = None

                inst_id_list = np.unique(instance_pred)[1:]  # exlcude background
                inst_info_dict = {}
                for inst_id in inst_id_list:
                    inst_map = instance_pred == inst_id
                    # TODO: chane format of bbox output
                    rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
                    inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
                    inst_map = inst_map[
                        inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
                    ]
                    inst_map = inst_map.astype(np.uint8)
                    inst_moment = cv2.moments(inst_map)
                    inst_contour = cv2.findContours(
                        inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )
                    # * opencv protocol format may break
                    inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
                    # < 3 points dont make a contour, so skip, likely artifact too
                    # as the contours obtained via approximation => too small or sthg
                    if inst_contour.shape[0] < 3:
                        continue
                    if len(inst_contour.shape) != 2:
                        continue # ! check for trickery shape
                    inst_centroid = [
                        (inst_moment["m10"] / inst_moment["m00"]),
                        (inst_moment["m01"] / inst_moment["m00"]),
                    ]
                    inst_centroid = np.array(inst_centroid)
                    inst_contour[:, 0] += inst_bbox[0][1]  # X
                    inst_contour[:, 1] += inst_bbox[0][0]  # Y
                    inst_centroid[0] += inst_bbox[0][1]  # X
                    inst_centroid[1] += inst_bbox[0][0]  # Y
                    inst_info_dict[inst_id] = {  # inst_id should start at 1
                        "bbox": inst_bbox,
                        "centroid": inst_centroid,
                        "contour": inst_contour,
                        "type_prob": None,
                        "type": None,
                    }

                #### * Get class of each instance id, stored at index id-1
                for inst_id in list(inst_info_dict.keys()):
                    rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
                    inst_map_crop = instance_pred[rmin:rmax, cmin:cmax]
                    inst_type_crop = mask_pred[rmin:rmax, cmin:cmax]
                    inst_map_crop = (
                        inst_map_crop == inst_id
                    )  # TODO: duplicated operation, may be expensive
                    inst_type = inst_type_crop[inst_map_crop]
                    type_list, type_pixels = np.unique(inst_type, return_counts=True)
                    type_list = list(zip(type_list, type_pixels))
                    type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                    inst_type = type_list[0][0]
                    if inst_type == 0:  # ! pick the 2nd most dominant if exist
                        if len(type_list) > 1:
                            inst_type = type_list[1][0]
                    type_dict = {v[0]: v[1] for v in type_list}
                    type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
                    inst_info_dict[inst_id]["type"] = int(inst_type)
                    inst_info_dict[inst_id]["type_prob"] = float(type_prob)

                nuc_val_list = list(inst_info_dict.values())
                # need singleton to make matlab happy
                nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
                nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
                nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])

                out_put = dict()
                out_put['inst_centroid'] = nuc_coms_list
                out_put['inst_type'] = nuc_type_list
                out_put['inst_map'] = instance_pred

                np.save(self.log_path/'pred'/f'{tissue_type[ins_idx]}.npy', out_put)





            metrics['IoU_score'] = iou_score.numpy()
            metrics['f1_score'] = f1_score.numpy()
            # metrics['f2_score'] = f2_score
            # metrics['accuracy'] = accuracy
            # metrics['recall'] = recall
            # metrics['Mean_IoU_Macro'] = macro_iou_score
            # metrics['f1_score_Macro'] = macro_f1_score

            for keys, values in metrics.items():
                metrics[keys] = np.round(values, 4)
                print(f'{keys} = {values}')
            print()
            #---->
            result = pd.DataFrame([metrics])
            result.to_csv(self.log_path / 'result.csv')

        else:
            print('Save logits')
            np.save(self.log_path / 'entropy.npy', logits.cpu().numpy())


    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)