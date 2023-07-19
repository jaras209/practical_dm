import json
from pathlib import Path

import numpy as np
import pandas as pd

from constants import ACTIONS

from metrics import compute_belief_state_metrics, compute_belief_state_exact_match_ratio
from huggingface_multiwoz_dataset import str_to_belief_state

path = Path('/Users/jaroslavsafar/Developer/HCN/data/action_supports.csv')


def check_actions(df):
    column_values = df.iloc[:, 0].tolist()
    dataframe_diff = set(column_values) - set(ACTIONS)
    actions_diff = set(ACTIONS) - set(column_values)
    return dataframe_diff, actions_diff

# Example usage
df = pd.read_csv(path)  # Replace 'your_data.csv' with your actual DataFrame file

dataframe_diff, actions_diff = check_actions(df)

if len(dataframe_diff) > 0:
    print("Actions in DataFrame that are not in the ACTIONS list:")
    print(dataframe_diff)
else:
    print("All actions in the DataFrame are in the ACTIONS list.")

if len(actions_diff) > 0:
    print("Actions in the ACTIONS list that are not in the DataFrame:")
    print(actions_diff)
else:
    print("All actions in the ACTIONS list are in the DataFrame.")