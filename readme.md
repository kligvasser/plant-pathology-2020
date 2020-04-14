# Plant Pathology 2020 - FGVC7

This respository contains my code for the [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) competition.

## Implemented

- [x] Basline starter 
- [ ] Effiecientnets two level classifier
- [ ] Consine with warm up learning rate
- [ ] Cutout
- [ ] Cutmix
- [ ] Lighting: PCA noising
- [ ] Test-time augmentations (TTA)
- [ ] OHEM loss
- [ ] Focal loss

## Scores

| Version | Notes | Score | LB |
| --- | --- | --- | --- |
| `2020-04-14_10-19-53` | basline, seed 5716 | 0.989 | 0.973 |


## Examples

Basline:
```shell
$ python3 main.py --model efficientnet --model-config "{'b_type': 3}" --root /home/tiras/datasets/kaggle/plants/ --crop-size 768 --use-tb --train-cross-validation --lr 0.0005 --step-size 10 --gamma 0.2 --epochs 20 --batch-size 16 --device-ids 0 1 2 3 --seed 5716
```
 
