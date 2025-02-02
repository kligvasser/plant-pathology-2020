import argparse
import torch
import logging
import signal
import sys
import torch.backends.cudnn as cudnn
from trainer import Trainer
from datetime import datetime
from os import path
from utils import misc
from random import randint

# torch.autograd.set_detect_anomaly(True)

def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch implementation for plant pathology 2020')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-ids', default=[0], type=int, nargs='+', help='device ids assignment (e.g 0 1 2 3')
    parser.add_argument('--model', default='resnet', help='model architecture (default: resnet)')
    parser.add_argument('--model-config', default='', help='additional architecture configuration')
    parser.add_argument('--file2load', default='', help='resume training from file (default: None)')
    parser.add_argument('--root', default='/home/kligtech/datasets/kaggle/plants', help='root dataset folder')
    parser.add_argument('--crop-size', default=768, type=int, help='image sizing (default: 768)')
    parser.add_argument('--crop-scale', default=1.1, type=float, help='crop scaling for evaluation (default: 1.1)')
    parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size (default: 16)')
    parser.add_argument('--epochs', default=10, type=int, help='epochs (default: 10)')
    parser.add_argument('--update-rate', default=1, type=int, help='update optimizer rate (default: 1)')
    parser.add_argument('--lr', default=5e-4, type=float, help='lr (default: 5e-4)')
    parser.add_argument('--step-size', default=5, type=int, help='scheduler step size (default: 5)')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler gamma (default: 0.5)')
    parser.add_argument('--exp-scheduler', default=False, action='store_true', help='use exp scheduler (default: false)')
    parser.add_argument('--weight-auc', default=1.0, type=float, help='weight for auc (default: 1.0)')
    parser.add_argument('--weight-acc', default=0., type=float, help='weight for acc (default: 0)')
    parser.add_argument('--tta', default=False, action='store_true', help='use test time transformations (default: false)')
    parser.add_argument('--cutmix', default=False, action='store_true', help='use cutmix transformations (default: false)')
    parser.add_argument('--num-splits', default=5, type=int, help='num of splits (default: 5)')
    parser.add_argument('--seed', default=-1, type=int, help='random seed (default: random)')
    parser.add_argument('--print-freq', default=10, type=int, help='print-freq (default: 10)')
    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
    parser.add_argument('--use-tb', default=False, action='store_true', help='use tensorboardx (default: false)')
    parser.add_argument('--train-cross-validation', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    # args.crop_size = tuple(args.crop_size)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save is '':
        args.save = time_stamp
    args.save_path = path.join(args.results_dir, args.save)
    if args.seed == -1:
        args.seed = randint(0, 12345)
    return args

def main():
    args = get_arguments()

    torch.manual_seed(args.seed)

    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # Set logs
    misc.mkdir(args.save_path)
    misc.setup_logging(path.join(args.save_path, 'log.txt'))

    # Print logs
    logging.info(args)

    # Trainer
    trainer = Trainer(args)
    if args.train_cross_validation:
        trainer.train_cross_validation()

    elif args.test:
        trainer.test()

    else:
        # trainer.train()
        trainer.train_no_eval()

if __name__ == '__main__':
    # Enables a ctrl-c without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    main()