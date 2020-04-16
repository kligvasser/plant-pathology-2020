from .datasets import Dataset, CutMix
from .transforms import get_transforms
from torch.utils.data import DataLoader

def get_loaders(args, df_train, df_eval):
    # Get transforms
    transforms = get_transforms(args)

    # Get datasets
    data_train = Dataset(root=args.root, df=df_train, transforms=transforms['train'])
    data_eval = Dataset(root=args.root, df=df_eval, transforms=transforms['eval'])

    if args.cutmix:
        data_train = CutMix(dataset=data_train)

    # Create dataloader
    loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    loader_eval = DataLoader(data_eval, batch_size=args.batch_size, shuffle=False, num_workers=4)

    loaders = {'train': loader_train, 'eval': loader_eval}
    return loaders