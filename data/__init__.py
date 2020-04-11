import albumentations as A
from albumentations.pytorch import ToTensorV2
from .datasets import Dataset
from torch.utils.data import DataLoader

def get_loaders(args, df_train, df_eval):
    # Get transforms
    transforms = get_transforms(args)

    # Get datasets
    data_train = Dataset(root=args.root, df=df_train, transforms=transforms['train'])
    data_eval = Dataset(root=args.root, df=df_eval, transforms=transforms['eval'])

    # Create dataloader
    loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    loader_eval = DataLoader(data_eval, batch_size=args.batch_size, shuffle=False, num_workers=4)

    loaders = {'train': loader_train, 'eval': loader_eval}
    return loaders

def get_transforms(args):
    # Train transforms
    transforms_train = A.Compose([
        A.RandomResizedCrop(height=args.crop_size, width=args.crop_size, p=1.0),
        A.Flip(),
        # A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
        # A.OneOf([
        #     A.IAAEmboss(p=1.0),
        #     A.IAASharpen(p=1.0),
        # ], p=0.5),
        # A.OneOf([
        #     A.ElasticTransform(p=1.0),
        #     A.IAAPiecewiseAffine(p=1.0),
        # ], p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ])

    # Eval transforms
    transforms_eval = A.Compose([
        A.Resize(height=args.crop_size, width=args.crop_size, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ])

    transforms = {'train': transforms_train, 'eval': transforms_eval}
    return transforms


