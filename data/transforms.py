import numpy as np
import torchvision

_IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

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

def get_transforms(args):
    # Train transforms
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(args.crop_size, scale=(0.2, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**_IMAGENET_STATS),
    ])

    # Eval transforms
    transforms_eval = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(args.crop_size * 1.1)),
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
    parser.add_argument('--crop-size', default=256, type=int)
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



