import torch
import os
import numpy as np
import cv2

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', df=None, training=True, transforms=None):
        self.root = root
        self.df = df
        self.training = training
        self.transforms = transforms

    def __getitem__(self, index):
        path = os.path.join(self.root, 'images', '{}.jpg'.format(self.df.loc[index, 'image_id']))

        input = cv2.imread(path, cv2.IMREAD_COLOR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        target = self.df.loc[index, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        target = torch.from_numpy(target.astype(np.int8))
        _, target = torch.max(target, dim=0)

        if self.transforms:
            input = self.transforms(image=input)['image']

        return {'input': input, 'target': target, 'path': path}

    def __len__(self):
        return self.df.shape[0]