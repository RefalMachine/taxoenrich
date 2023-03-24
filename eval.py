from taxoenrich.utils import *
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path')
    parser.add_argument('--reference_path')
    args = parser.parse_args()

    submitted = read_dataset(args.predict_path)
    reference = read_dataset(args.reference_path, json.loads)
    map_score, mrr_score = get_score(reference, submitted)
    print(f'Results for {args.predict_path}:\n\tMAP = {map_score}\n\tMRR = {mrr_score}')