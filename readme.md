# Plant Pathology 2020 - FGVC7

This repository contains my code for the [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) competition.

## Tested solutions

- [x] Baseline: efficient-b3 768x768
- [x] Architecture: two stages classifier
- [x] Architecture: efficient-b5
- [x] Architecture: efficient-b7
- [x] Learning rate: warm up exponential
- [x] Learning rate: warm up cosine
- [x] Augmentation: cutout ([paper](https://arxiv.org/pdf/1708.04552))
- [x] Augmentation: cutmix ([paper](https://arxiv.org/pdf/1905.04899.pdf))
- [x] Augmentation: PCA color ([paper](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf))
- [x] Augmentation: test-time augmentations (TTA)
- [x] Loss: weighted evaluation score
- [x] Loss: dense cross entropy
- [ ] Loss: OHEM ([paper](https://arxiv.org/pdf/1604.03540.pdf))
- [x] Loss: focal ([paper](https://arxiv.org/pdf/1708.02002.pdf))
- [ ] Loss: additive angular margin (ArcFace) ([paper](https://arxiv.org/pdf/1801.07698.pdf))
- [x] Optimization: self-training with noisy student ([paper](https://arxiv.org/pdf/1911.04252.pdf))

## Scores

| Version | Notes | Score | LB |
| :---: | :---: | :---: | :---: |
| `2020-04-14_10-19-53` | baseline | 0.989 | 0.973 |
| `2020-04-14_17-10-46` | exponential lr | 0.988 | 0.969 |
| `2020-04-14_20-53-54` | cosine lr | 0.988 | 0.969 |
| `2020-04-15_01-01-48` | focal loss | 0.988 | 0.971 |
| `2020-04-15_07-52-47` | cutout and lighting | 0.989 | 0.971 |
| `2020-04-15_07-52-47` | two stages classifier | 0.988 | 0.971 |
| `2020-04-15_14-31-43` | inception augmentations | 0.988 | 0.973 |
| `2020-04-15_21-11-49` | 500x750 |  0.987 | 0.970 |
| `2020-04-16_10-23-15` | weighted: 0.9 & 0.1 | 0.985 | 0.975 |
| `2020-04-16_11-50-25` | weighted: 0.8 & 0.2 | 0.982 | 0.973 |
| `2020-04-16_15-57-40` | cutmix | 0.985 | 0.961 |
| `2020-04-16_18-44-52` | dense cross entropy | 0.989 | **0.976** |
| `2020-04-17_15-34-48` | test-time augmentations | 0.989 | 0.971 |

## Examples

Baseline (LB 0.973):
```shell
$ python3 main.py --model efficientnet --model-config "{'b_type': 3}" --root /home/tiras/datasets/kaggle/plants/ --crop-size 768 --use-tb --train-cross-validation --lr 0.0005 --step-size 10 --gamma 0.2 --epochs 20 --batch-size 16 --device-ids 0 1 2 3 --seed 5716
```
 
 Weighted (LB 0.975):
 ```shell
 $ python3 main.py --model efficientnet --model-config "{'b_type': 3}" --root /home/tiras/datasets/kaggle/plants/ --crop-size 768 --use-tb --train-cross-validation --lr 0.0005 --step-size 10 --gamma 0.2 --epochs 20 --batch-size 16 --device-ids 0 1 2 3 --seed 5716 --weight-auc 0.9 --weight-acc 0.1
```
