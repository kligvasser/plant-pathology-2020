from .datasets import Dataset, CutMix
from .transforms import get_transforms
from torch.utils.data import DataLoader

def get_loaders(args, df_train, df_eval):
    # Get transforms
    transforms = get_transforms(args)

    # Train loader
    data_train = Dataset(root=args.root, df=df_train, transforms=transforms['train'])

    if args.cutmix:
        data_train = CutMix(dataset=data_train)

    loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Eval loader
    data_eval = Dataset(root=args.root, df=df_eval, transforms=transforms['eval'])
    loader_eval = DataLoader(data_eval, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Test loader
    if args.tta:
        data_test = Dataset(root=args.root, df=df_eval, transforms=transforms['test'])
        batch_size = max(1, int(args.batch_size ** 0.5))
    else:
        data_test = Dataset(root=args.root, df=df_eval, transforms=transforms['eval'])
        batch_size = args.batch_size

    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=4)

    loaders = {'train': loader_train, 'eval': loader_eval, 'test': loader_test}
    return loaders