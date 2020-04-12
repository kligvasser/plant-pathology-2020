import torch
import os
import numpy as np
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
        target = torch.from_numpy(target.astype(np.int8))
        _, target = torch.max(target, dim=0)

        if self.transforms:
            input = self.transforms(input)

        return {'input': input, 'target': target, 'path': path}

    def __len__(self):
        return self.df.shape[0]