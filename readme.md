# Plant Pathology 2020 - FGVC7

This repository contains my code for the [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) competition.

## Tested solutions

- [x] Baseline starter 
- [x] Efficient-nets two stages classifier
- [ ] Efficient-b5 and 480x768
- [x] Exponential with warm up learning rate
- [x] Cosine with warm up learning rate
- [x] Cutout
- [ ] Cutmix
- [x] Lighting: PCA noising
- [ ] OHEM loss
- [x] Focal loss
- [ ] Additive Angular Margin Loss (ArcFace)
- [ ] Test-time augmentations (TTA)

## Scores

| Version | Notes | Score | LB |
| --- | --- | --- | --- |
| `2020-04-14_10-19-53` | baseline | 0.989 | 0.973 |
| `2020-04-14_17-10-46` | exponential lr | 0.988 | 0.969 |
| `2020-04-14_20-53-54` | cosine lr | 0.988 | 0.969 |
| `2020-04-15_01-01-48` | focal loss | 0.988 | 0.971 |
| `2020-04-15_07-52-47` | cutout and lighting | 0.989 | 0.971 |
| `2020-04-15_07-52-47` | two stages classifier | 0.988 | 0.971 |
| `2020-04-15_14-31-43` | inception augmentations | 0.988 | 0.973 |

## Examples

Baseline:
```shell
$ python3 main.py --model efficientnet --model-config "{'b_type': 3}" --root /home/tiras/datasets/kaggle/plants/ --crop-size 768 --use-tb --train-cross-validation --lr 0.0005 --step-size 10 --gamma 0.2 --epochs 20 --batch-size 16 --device-ids 0 1 2 3 --seed 5716
```
 
