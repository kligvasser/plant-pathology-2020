import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import numbers
import random
from PIL import Image


_IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

class CutOut(object):
    def __init__(self, n_holes=1, length=128):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, image):
        h = image.size(1)
        w = image.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask
        return image

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

class NRandomCrop(object):
    def __init__(self, size, n=1, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.n = n

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for i in range(n)]
        j_list = [random.randint(0, w - tw) for i in range(n)]
        return i_list, j_list, th, tw

    def __call__(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size, self.n)

        return n_random_crops(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def n_random_crops(img, x, y, h, w):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    crops = []
    for i in range(len(x)):
        new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
        crops.append(new_crop)
    return tuple(crops)

def get_transforms(args):
    # Training transforms
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(args.crop_size, scale=(0.2, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.ToTensor(),
        Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
        torchvision.transforms.Normalize(**_IMAGENET_STATS),
    ])

    # Evaluation transforms
    scale_size = int(min(args.crop_size) * args.crop_scale) if isinstance(args.crop_size, tuple) else int(args.crop_size * args.crop_scale)
    transforms_eval = torchvision.transforms.Compose([
        torchvision.transforms.Resize(scale_size),
        torchvision.transforms.CenterCrop(args.crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**_IMAGENET_STATS),
    ])

    # Test transforms
    if args.tta:
        transforms_tta = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(**_IMAGENET_STATS),
            ])

        transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(scale_size),
            torchvision.transforms.TenCrop(args.crop_size),
            torchvision.transforms.Lambda(lambda crops: torch.stack([transforms_tta(crop) for crop in crops]))
            ])
    else:
        transforms_test = None

    transforms = {'train': transforms_train, 'eval': transforms_eval, 'test': transforms_test}
    return transforms

if __name__ == "__main__":
    import argparse
    import sys, os
    import random
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    from utils.misc import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--crop-size', default=500, type=int, help='image sizing (default: 500)')
    parser.add_argument('--crop-scale', default=1.1, type=float, help='crop scaling for evaluation (default: 1.1)')
    parser.add_argument('--tta', default=16, type=int, help='tta num (default: 16)')
    parser.add_argument('--root', default='/home/kligtech/datasets/kaggle/plants', help='root dataset folder')
    args = parser.parse_args()

    transforms = get_transforms(args)
    input = Image.open(os.path.join(args.root, 'images', 'Train_{}.jpg'.format(random.randint(0, 1000)))).convert('RGB')

    # Eval
    plot_image_grid(transforms['eval'](input), 1)

    # Train
    images = None
    for i in range(16):
        image = (transforms['train'](input)).unsqueeze(dim=0)
        if images is None:
            images = image
        else:
            images = torch.cat((images, image), dim=0)

    plot_image_grid(images, 4)

    # Test
    images = transforms['test'](input)
    ncrops, c, h, w = images.size()
    plot_image_grid(images.view(-1, c, h, w), 4)



