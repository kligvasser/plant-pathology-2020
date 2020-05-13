import argparse
import glob
import os
import pandas as pd
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description='Average results')
    parser.add_argument('--root', default='./csv/submissions/', required=False, help='csv root folder')
    parser.add_argument('--method', default='mean', choices=['mean', 'l1', 'l2'], required=False, help='method of averaging')
    args = parser.parse_args()
    return args

def get_csv_list(args):
    csv_list = glob.glob(os.path.join(args.root, '*.csv'))
    return csv_list

def write_csv(args, preds, csv_file):
    df = pd.read_csv(csv_file)
    df[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds
    df.to_csv(os.path.join(args.root, 'averaged_{}.csv'.format(args.method)), index=False)

def main():
    args = get_arguments()

    csv_list = get_csv_list(args)
    csvs = np.stack([pd.read_csv(csv).iloc[:, 1:].values for csv in csv_list], axis=0)

    if args.method == 'l1':
        weights = np.abs(csvs - 0.5)
        preds = np.average(csvs, axis=0, weights=weights)
        preds = preds / preds.sum(axis=1, keepdims=True)
    elif args.method == 'l2':
        weights = (csvs - 0.5) ** 2
        preds = np.average(csvs, axis=0, weights=weights)
        preds = preds / preds.sum(axis=1, keepdims=True)
    else:
        preds = np.mean(csvs, axis=0)

    write_csv(args, preds, csv_list[0])

if __name__ == "__main__":
    main()


