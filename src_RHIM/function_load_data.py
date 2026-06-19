import pandas as pd
import argparse
from pprint import pprint

def read_argument():
    args = argparse.ArgumentParser()
    args.add_argument('-f', '--file_path')
    args.add_argument('-col', '--column')
    args = args.parse_args()
    return args

def load_dataframe(file_path):
    data = pd.read_excel(file_path)
    return data

def load_concerned_column(data, column):
    return data[column].tolist()

def main():
    args = read_argument()
    data = load_dataframe(args.file_path) 
    concol = load_concerned_column(data, args.column)
    pprint(concol)

if __name__ == '__main__':
    main()
