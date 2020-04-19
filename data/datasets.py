import torch
import os
import numpy as np
import random
from PIL import Image

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', df=None, training=True, transforms=None):
        self.root = root
        self.df = df
        self.training = training
        self.transforms = transforms

    def __getitem__(self, index):
        path = os.path.join(self.root, 'images', '{}.jpg'.format(self.df.loc[index, 'image_id']))

        input = Image.open(path).convert('RGB')

        target = self.df.loc[index, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        target = torch.from_numpy(target.astype(np.float))
        _, target = torch.max(target, dim=0)

        if self.transforms:
            input = self.transforms(input)

        return {'input': input, 'target': target, 'path': path}

    def __len__(self):
        return self.df.shape[0]

class CutMix(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, num_classes=4, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        data = self.dataset[index]
        img, lb, pth = data['input'], data['target'], data['path']

        lb_onehot = self._onehot(lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            data = self.dataset[rand_index]
            img2, lb2 = data['input'], data['target']
            lb2_onehot = self._onehot(lb2)

            bbx1, bby1, bbx2, bby2 = self._rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return {'input': img, 'target': lb_onehot, 'path': pth}

    def __len__(self):
        return len(self.dataset)

    def _onehot(self, target):
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        vec[target] = 1.
        return vec

    def _rand_bbox(self, size, lam):
        if len(size) == 4:
            W = size[2]
            H = size[3]
        elif len(size) == 3:
            W = size[1]
            H = size[2]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2