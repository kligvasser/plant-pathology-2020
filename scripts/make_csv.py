import argparse
import pandas as pd
import numpy as np

_CLASS_PROPABILITY = [0.28, 0.05, 0.35, 0.32]

def get_arguments():
    parser = argparse.ArgumentParser(description='Generate csv file for noisy student')
    parser.add_argument('--csv-file', default='', required=True, help='csv input file')
    parser.add_argument('--num-samples', default=800, required=False, help='number of samples (default: 800)')
    parser.add_argument('--margin', default=0.5, required=False, help='margin (default: 0.5)')

    args = parser.parse_args()
    return args

def read_csv(args):
    df = pd.read_csv(args.csv_file)
    arr = df.iloc[:, 1:].values
    indices = np.sort(arr)[:, ::-1]
    indices = indices[:, 0] - indices[:, 1] > args.margin
    df = df[indices].round()
    return df

def sample_df(df, args):
    frames = []
    for i, k in enumerate(df.keys()[1:]):
        frames.append(df[df[k] == 1.].sample(n=int(args.num_samples * _CLASS_PROPABILITY[i])))
    sampled = pd.concat(frames).sort_index()
    return sampled

def save_df(df, args):
    df.to_csv('./csv/generated_{}.csv'.format(args.num_samples), index=False)

def main():
    args = get_arguments()
    df = read_csv(args)
    df = sample_df(df, args)
    save_df(df, args)

if __name__ == "__main__":
    main()


