# Plant Pathology 2020 - FGVC7

This repository contains my code for the [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) competition.

## Implemented

- [x] Baseline starter 
- [ ] Efficient-nets two level classifier
- [x] Exponential with warm up learning rate
- [ ] Cosine with warm up learning rate
- [ ] Cutout
- [ ] Cutmix
- [ ] Lighting: PCA noising
- [ ] Test-time augmentations (TTA)
- [ ] OHEM loss
- [ ] Focal loss
- [ ] Additive Angular Margin Loss (ArcFace)

## Scores

| Version | Notes | Score | LB |
| --- | --- | --- | --- |
| `2020-04-14_10-19-53` | baseline | 0.989 | 0.973 |
| `2020-04-14_17-10-46` | exponential lr 20e | 0.988 | 0.969 |
| `2020-04-14_18-35-56` | exponential lr 25e | 0.990 |  |


## Examples

Baseline:
```shell
$ python3 main.py --model efficientnet --model-config "{'b_type': 3}" --root /home/tiras/datasets/kaggle/plants/ --crop-size 768 --use-tb --train-cross-validation --lr 0.0005 --step-size 10 --gamma 0.2 --epochs 20 --batch-size 16 --device-ids 0 1 2 3 --seed 5716
```
 
