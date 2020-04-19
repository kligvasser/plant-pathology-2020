import argparse
import random
import os

def get_arguments():
    parser = argparse.ArgumentParser(description='Running with random settings')
    parser.add_argument('--root', default='', required=True, help='dataset root folder')
    parser.add_argument('--iterations', default=10, type=int, help='number of iterations (default 10)')
    args = parser.parse_args()
    return args

def rand_flag(flag, p=0.5):
    r = random.random()
    if r > p:
        return flag
    else:
        return ''

def get_params():
    crop_size = 768
    crop_scale = random.choice([1.1, 1.15, 1.2])
    lr = random.uniform(0.0003, 0.00055)
    step_size = random.randint(4, 8)
    epochs = random.randint(8, 15)
    seed = random.randint(1, 10000)
    auc = random.uniform(0.85, 1.0)
    acc = 1 - auc

    return {'crop_size': crop_size, 'crop_scale': crop_scale, 'lr': lr, 'step_size': step_size, 'epochs': epochs, 'seed': seed, 'auc': auc, 'acc': acc}

def set_cmd(args, params):
    cmd = 'python3 main.py --model efficientnet --root {} --crop-size {} --crop-scale {} --train-cross-validation --lr {:.6f} --step-size {} --gamma 0.2 --epochs {} --batch-size 16 --device-ids 0 1 2 3 --seed {} --weight-auc {:.2f} --weight-acc {:.2f}'.format(args.root, params['crop_size'], params['crop_scale'], params['lr'], params['step_size'], params['epochs'], params['seed'], params['auc'], params['acc'])
    return cmd

def main():
    args = get_arguments()

    for i in range(args.iterations):
        params = get_params()
        cmd = set_cmd(args, params)
        print('\n\nRunning {}/{}: {}'.format(i + 1, args.iterations, cmd))
        os.system(cmd)

if __name__ == "__main__":
    main()


