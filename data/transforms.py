import torch
import numpy as np
import torchvision

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

def get_transforms(args):
    # Training transforms
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(args.crop_size, scale=(0.2, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**_IMAGENET_STATS),
    ])

    # Evaluation transforms
    scale_size = int(min(args.crop_size) * 1.1) if isinstance(args.crop_size, tuple) else int(args.crop_size * 1.1)
    transforms_eval = torchvision.transforms.Compose([
        torchvision.transforms.Resize(scale_size),
        torchvision.transforms.CenterCrop(args.crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**_IMAGENET_STATS),
    ])

    transforms = {'train': transforms_train, 'eval': transforms_eval}
    return transforms

if __name__ == "__main__":
    import argparse
    import sys, os
    import random
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    from utils.misc import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--crop-size', default=500, type=int, nargs=2, help='image sizing (default: 500)')
    args = parser.parse_args()

    transforms = get_transforms(args)
    input = Image.open('/home/kligtech/datasets/plants/images/Train_{}.jpg'.format(random.randint(0, 1000))).convert('RGB')

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



