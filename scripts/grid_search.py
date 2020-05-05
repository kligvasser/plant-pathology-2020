import argparse
import random
import os

def get_arguments():
    parser = argparse.ArgumentParser(description='Running with random settings')
    parser.add_argument('--root', default='', required=True, help='dataset root folder')
    parser.add_argument('--iterations', default=10, type=int, help='number of iterations (default: 10)')
    args = parser.parse_args()
    return args

def rand_flag(flag, p=0.5):
    r = random.random()
    if r < p:
        return flag
    else:
        return ''

def get_params_efficient():
    crop_size = 768
    crop_scale = 1.1
    lr = random.uniform(0.00041, 0.00044)
    step_size = random.randint(7, 8)
    epochs = random.randint(16, 18)
    seed = random.randint(1, 12345)
    auc = random.uniform(0.95, 1.0)
    acc = 1. - auc
    splits = random.randint(4, 7)
    tta = rand_flag('--tta', 0.25)

    return {'crop_size': crop_size, 'crop_scale': crop_scale, 'lr': lr, 'step_size': step_size, 'epochs': epochs, 'seed': seed, 'auc': auc, 'acc': acc, 'tta': tta, 'splits': splits}

def get_params_inception():
    crop_size = 768
    crop_scale = 1.1
    lr = random.uniform(0.00040, 0.00045)
    step_size = random.randint(9, 11)
    epochs = random.randint(18, 22)
    seed = random.randint(1, 12345)
    auc = random.uniform(0.95, 1.0)
    acc = 1. - auc
    splits = random.randint(4, 6)
    tta = rand_flag('--tta', 0.25)

    return {'crop_size': crop_size, 'crop_scale': crop_scale, 'lr': lr, 'step_size': step_size, 'epochs': epochs, 'seed': seed, 'auc': auc, 'acc': acc, 'tta': tta, 'splits': splits}

def set_cmd_efficient(args, params):
    cmd = 'python3 main.py --model efficientnet --root {} --crop-size {} --crop-scale {} --train-cross-validation --lr {:.6f} --step-size {} --gamma 0.2 --epochs {} --batch-size 16 --device-ids 0 1 2 3 --seed {} --weight-auc {:.2f} --weight-acc {:.2f} --num-splits {} {}'.format(args.root, params['crop_size'], params['crop_scale'], params['lr'], params['step_size'], params['epochs'], params['seed'], params['auc'], params['acc'], params['splits'], params['tta'])
    return cmd

def set_cmd_inception(args, params):
    cmd = 'python3 main.py --model inception --root {} --crop-size {} --crop-scale {} --train-cross-validation --lr {:.6f} --step-size {} --gamma 0.2 --epochs {} --batch-size 32 --device-ids 0 1 2 3 --seed {} --weight-auc {:.2f} --weight-acc {:.2f} --num-splits {} {}'.format(args.root, params['crop_size'], params['crop_scale'], params['lr'], params['step_size'], params['epochs'], params['seed'], params['auc'], params['acc'], params['splits'], params['tta'])
    return cmd

def main():
    args = get_arguments()

    for i in range(args.iterations):
        params = get_params_inception()
        cmd = set_cmd_inception(args, params)
        print('\n\nRunning {}/{}: {}'.format(i + 1, args.iterations, cmd))
        os.system(cmd)

if __name__ == "__main__":
    main()


