The structure of this code is:
- DTSeg
    - DTSeg
        - diffusion_encoder
        - transformer_decoder
        
    - Semi-supervised_methods
        

``DTSeg`` includes the code for training DTSeg and collaborative learning. ``diffusion_encoder`` includes the code for the unsupervised training of the latent diffusion model. ``transformer_decoder`` includes the code for feature extraction by the latent diffusion model, training the transformer-based decoder, and collaborative learning. 
``Semi-supervised_methods`` includes the code for all comparative semi-supervised methods, including the adversarial network, cross pseudo-supervision method, deep co-training method, and uncertainty-aware mean teacher method. 

## Environment construction
Please follow the README.md in latent diffusion model
```DTSeg/diffusion_encoder/README.md```.

## Training of latent diffusion model
### Diffusion (Big)
```python
CUDA_VISIBLE_DEVICES=0 python DTSeg/diffusion_encoder/main.py --base DTSeg/diffusion_encoder/configs/latent-diffusion/big.yaml -t
```
### Diffusion (PanNuke)
```python
CUDA_VISIBLE_DEVICES=0 python DTSeg/diffusion_encoder/main.py --base DTSeg/diffusion_encoder/configs/latent-diffusion/pannuke.yaml -t
```
### Diffusion (MoNuSAC)
```python
CUDA_VISIBLE_DEVICES=0 python DTSeg/diffusion_encoder/main.py --base DTSeg/diffusion_encoder/configs/latent-diffusion/monusac.yaml -t
```

## Training of transformer-based decoder with Diffusion (Big)
### MoNuSAC seed (42, 107, 412)
```python
for Model in DTSeg_0.05_789_aug_big DTSeg_0.1_789_aug_big DTSeg_0.2_789_aug_big DTSeg_1.0_789_aug_big
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done

for Model in DTSeg_0.05_789_aug_big DTSeg_0.1_789_aug_big DTSeg_0.2_789_aug_big DTSeg_1.0_789_aug_big
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac_107/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac_107/$Model.yaml --gpus 0
done

for Model in DTSeg_0.05_789_aug_big DTSeg_0.1_789_aug_big DTSeg_0.2_789_aug_big DTSeg_1.0_789_aug_big
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac_412/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac_412/$Model.yaml --gpus 0
done
```

## Training of transformer-based decoder with Diffusion (PanNuke/MoNuSAC), limited pre-training data, OOD cases
### MoNuSAC seed (42, 107, 412)
```python
for Model in DTSeg_0.05_789_aug_pannuke DTSeg_0.1_789_aug_pannuke DTSeg_0.2_789_aug_pannuke
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done

for Model in DTSeg_0.05_789_aug_pannuke DTSeg_0.1_789_aug_pannuke DTSeg_0.2_789_aug_pannuke
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac_107/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac_107/$Model.yaml --gpus 0
done

for Model in DTSeg_0.05_789_aug_pannuke DTSeg_0.1_789_aug_pannuke DTSeg_0.2_789_aug_pannuke
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac_412/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac_412/$Model.yaml --gpus 0
done
```

## Training of supervised baseline model
```python
for Model in FPN_0.05_aug FPN_0.1_aug FPN_0.2_aug FPN_1.0_aug
    do
    CUDA_VISIBLE_DEVICES=0 python train_image.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_image.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done

for Model in FPN_0.05_aug FPN_0.1_aug FPN_0.2_aug FPN_1.0_aug
    do
    CUDA_VISIBLE_DEVICES=0 python train_image.py --stage train --config DTSeg/transformer_decoder/scripts/monusac_107/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_image.py --stage test --config DTSeg/transformer_decoder/scripts/monusac_107/$Model.yaml --gpus 0
done

for Model in FPN_0.05_aug FPN_0.1_aug FPN_0.2_aug FPN_1.0_aug
    do
    CUDA_VISIBLE_DEVICES=0 python train_image.py --stage train --config DTSeg/transformer_decoder/scripts/monusac_412/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_image.py --stage test --config DTSeg/transformer_decoder/scripts/monusac_412/$Model.yaml --gpus 0
done
```

## Training of collaborative learning with Diffusion (PanNuke/MoNuSAC), limited pre-training data, OOD cases
### MoNuSAC seed (42, 107, 412)
```python
for Model in aggregrate_789_fpn_0.05_wsa_pannuke aggregrate_789_fpn_0.1_wsa_pannuke aggregrate_789_fpn_0.2_wsa_pannuke
    do
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done

for Model in aggregrate_789_fpn_0.05_wsa_pannuke aggregrate_789_fpn_0.1_wsa_pannuke aggregrate_789_fpn_0.2_wsa_pannuke
    do
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage train --config DTSeg/transformer_decoder/scripts/monusac_107/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage test --config DTSeg/transformer_decoder/scripts/monusac_107/$Model.yaml --gpus 0
done

for Model in aggregrate_789_fpn_0.05_wsa_pannuke aggregrate_789_fpn_0.1_wsa_pannuke aggregrate_789_fpn_0.2_wsa_pannuke
    do
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage train --config DTSeg/transformer_decoder/scripts/monusac_412/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage test --config DTSeg/transformer_decoder/scripts/monusac_412/$Model.yaml --gpus 0
done
```

## Training of comparative semi-supervised methods, for example, cross pseudo supervision method
### MoNuSAC, for example, seed 42
```python
# ### monusac fpn 42
CUDA_VISIBLE_DEVICES=6 python train_cross_pseudo_supervision_2D.py --root_path ../data/MoNuSAC_42 --csv_path monusac_42.csv --label_ratio 0.05 --model fpn --num_classes 5 --exp MoNuSAC_42/Cross_Pseudo_Supervision --pretrained /data114_3/shaozc/data111_shaozc/shaozc/SegDiff/SegDiff/logs/scripts/monusac/FPN_0.05_aug/epoch=76-val_IoU=0.6252.ckpt
CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/MoNuSAC_42 --csv_path monusac_42.csv --label_ratio 0.05 --model fpn --num_classes 5 --exp MoNuSAC_42/Cross_Pseudo_Supervision

CUDA_VISIBLE_DEVICES=6 python train_cross_pseudo_supervision_2D.py --root_path ../data/MoNuSAC_42 --csv_path monusac_42.csv --label_ratio 0.1 --model fpn --num_classes 5 --exp MoNuSAC_42/Cross_Pseudo_Supervision --pretrained /data114_3/shaozc/data111_shaozc/shaozc/SegDiff/SegDiff/logs/scripts/monusac/FPN_0.1_aug/epoch=121-val_IoU=0.6333.ckpt
CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/MoNuSAC_42 --csv_path monusac_42.csv --label_ratio 0.1 --model fpn --num_classes 5 --exp MoNuSAC_42/Cross_Pseudo_Supervision

CUDA_VISIBLE_DEVICES=6 python train_cross_pseudo_supervision_2D.py --root_path ../data/MoNuSAC_42 --csv_path monusac_42.csv --label_ratio 0.2 --model fpn --num_classes 5 --exp MoNuSAC_42/Cross_Pseudo_Supervision --pretrained /data114_3/shaozc/data111_shaozc/shaozc/SegDiff/SegDiff/logs/scripts/monusac/FPN_0.2_aug/epoch=70-val_IoU=0.6463.ckpt
CUDA_VISIBLE_DEVICES=6 python Mytest.py --root_path ../data/MoNuSAC_42 --csv_path monusac_42.csv --label_ratio 0.2 --model fpn --num_classes 5 --exp MoNuSAC_42/Cross_Pseudo_Supervision

```

## Ablation study

### 1. Self-attention, for example, seed 42
```python
### Diffusion (Big)
for Model in DTSeg_0.05_789_aug_big_wsa DTSeg_0.1_789_aug_big_wsa DTSeg_0.2_789_aug_big_wsa
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done
### Diffusion (PanNuke)
for Model in DTSeg_0.05_789_aug_pannuke_wsa DTSeg_0.1_789_aug_pannuke_wsa DTSeg_0.2_789_aug_pannuke_wsa
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done
### Diffusion (MoNuSAC)
for Model in DTSeg_0.05_789_aug_monusac_wsa DTSeg_0.1_789_aug_monusac_wsa DTSeg_0.2_789_aug_monusac_wsa
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done
```

### 2. Series vs. Parallel, for example, seed 42
```python
### Diffusion (Big)
for Model in DTSeg_0.05_789_aug_big_series DTSeg_0.1_789_aug_big_series DTSeg_0.2_789_aug_big_series
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done
### Diffusion (PanNuke)
for Model in DTSeg_0.05_789_aug_pannuke_series DTSeg_0.1_789_aug_pannuke_series DTSeg_0.2_789_aug_pannuke_series
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done
### Diffusion (MoNuSAC)
for Model in DTSeg_0.05_789_aug_monusac_series DTSeg_0.1_789_aug_monusac_series DTSeg_0.2_789_aug_monusac_series
    do
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_pro.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done
```

### 3. Different collaborators, for example, seed 42

```python
### fpn+fpn
for Model in aggregrate_fpn_fpn_0.05.yaml aggregrate_fpn_fpn_0.1.yaml aggregrate_fpn_fpn_0.2.yaml
    do
    CUDA_VISIBLE_DEVICES=0 python train_both_image.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_both_image.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done

### dtseg+dtseg
for Model in aggregrate_789_789_0.05_big aggregrate_789_789_0.1_big aggregrate_789_789_0.2_big
    do
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done

### dtseg+fpn
for Model in aggregrate_789_fpn_0.05_wsa_big aggregrate_789_fpn_0.1_wsa_big aggregrate_789_fpn_0.2_wsa_big 
    do
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage train --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
    CUDA_VISIBLE_DEVICES=0 python train_both.py --stage test --config DTSeg/transformer_decoder/scripts/monusac/$Model.yaml --gpus 0
done
```