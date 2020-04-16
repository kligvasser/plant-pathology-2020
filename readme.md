# Plant Pathology 2020 - FGVC7

This repository contains my code for the [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) competition.

## Tested solutions

- [x] Baseline: efficient-b3 
- [x] Architecture: efficient-nets two stages classifier
- [x] Architecture: efficient-b5 and 500x750
- [x] Learning rate: warm up exponential
- [x] Learning rate: warm up cosine
- [x] Augmentation: cutout
- [ ] Augmentation: cutmix ([paper](https://arxiv.org/pdf/1905.04899.pdf))
- [x] Augmentation: PCA color ([paper](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf))
- [ ] Augmentation: test-time augmentations (TTA)
- [ ] Loss: weighted evaluation score
- [ ] Loss: OHEM ([paper](https://arxiv.org/pdf/1604.03540.pdf))
- [x] Loss: focal ([paper](https://arxiv.org/pdf/1708.02002.pdf))
- [ ] Loss: additive angular margin (ArcFace) ([paper](https://arxiv.org/pdf/1801.07698.pdf))

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

## Examples

Baseline:
```shell
$ python3 main.py --model efficientnet --model-config "{'b_type': 3}" --root /home/tiras/datasets/kaggle/plants/ --crop-size 768 --use-tb --train-cross-validation --lr 0.0005 --step-size 10 --gamma 0.2 --epochs 20 --batch-size 16 --device-ids 0 1 2 3 --seed 5716
```
 
