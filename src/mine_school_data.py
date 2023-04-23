import json

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse

DATASET_DIR = '../data/school_data/tracker.json'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=DATASET_DIR, help='')

    return parser.parse_args()


def mine_data(args):
    with open(args.data_path, 'r') as file:
        loaded_data = json.load(file)

    default_keys = loaded_data[0].keys()
    all_data = {key: [] for key in default_keys}
    for item in loaded_data:
        print(item)
        for key, value in item.items():
            all_data[key].append(value)

    unique = {key: set(value) for key, value in all_data.items() if key != 'data'}

    data = dict()
    keyyys = set()
    ddd = set()
    bot = []
    user = []
    text = set()
    for item in all_data['data']:
        if item['event'] not in ['user_featurization', 'followup', 'action']:
            if item['event'] == 'bot':
                bot.append(item)
            if item['event'] == 'user':
                user.append(item)
                text.add(item['text'])

        for key, value in item.items():
            if key in data.keys():
                data[key].append(value)

            else:
                data[key] = [value]

    text_series = pd.Series(list(text))
    print(text_series)
    text_series.to_csv("user_text.csv")

    for item in bot:
        print(item)
    """
    print(f"{'=' * 30}\nBOT\n")
    for item in bot:
        print(item)

    print(f"{'=' * 30}\nUSER\n")
    for item in user:
        print(item)

    print(set(data['event']))
    """


if __name__ == '__main__':
    mine_data(parse_args())
