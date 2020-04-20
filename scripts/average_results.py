import argparse
import glob
import os
import pandas as pd

def get_arguments():
    parser = argparse.ArgumentParser(description='Average results')
    parser.add_argument('--root', default='./csv/submissions/', required=False, help='csv root folder')
    args = parser.parse_args()
    return args

def get_csv_list(args):
    csv_list = glob.glob(os.path.join(args.root, '*.csv'))
    return csv_list

def write_csv(args, preds, csv_file):
    df = pd.read_csv(csv_file)
    df[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds
    df.to_csv(os.path.join(args.root, 'averaged.csv'), index=False)

def main():
    args = get_arguments()

    csv_list = get_csv_list(args)

    preds = None
    for csv_file in csv_list:
        pred = pd.read_csv(csv_file).iloc[:, 1:].values

        if preds is None:
            preds = pred / len(csv_list)
        else:
            preds += pred / len(csv_list)

    write_csv(args, preds, csv_list[0])

if __name__ == "__main__":
    main()


