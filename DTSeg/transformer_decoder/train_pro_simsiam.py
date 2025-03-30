import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models.model_interface_pro_simsiam import ModelInterfaceProSimsiam
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='scripts/monusac/UNetFormer_0.1_simsiam_big.yaml',type=str)
    parser.add_argument('--gpus', default = [7])
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,
                'state': cfg.General.server}
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'model_feature': cfg.Model_Feature,
                            'model_image': cfg.Model_Image,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'state': cfg.General.server
                            }
    model = ModelInterfaceProSimsiam(**ModelInterface_dict)

    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        gpus=cfg.General.gpus,
        amp_level=cfg.General.amp_level,  
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
        # limit_train_batches=0.01,
        # limit_val_batches=0.01,
        # limit_test_batches=0.01,
    )

    #---->train or test
    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            pretrained_dict = torch.load(path, map_location='cpu')['model_state_dict']
            # pretrained_dict = {k for k, v in checkpoint['model_state_dict'].items()}
            for k, v in model.model.state_dict().items():
                if k not in pretrained_dict:
                    print(f'key "{k}" could not be found in loaded state dict')
                elif pretrained_dict[k].shape != v.shape:
                    print(f'key "{k}" is of different shape in model and loaded state dict')
                    pretrained_dict[k] = v
            msg = model.model.load_state_dict(pretrained_dict, strict=False) #feature extractor do not need to load the weight
            model.to(device)
            print(f'loaded pretrained model with msg: {msg}')
            trainer.test(model=model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage

    #---->main
    main(cfg)
 