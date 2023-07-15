import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import compute_belief_state_metrics, compute_belief_state_exact_match_ratio
from huggingface_multiwoz_dataset import str_to_belief_state

path = Path('/Users/jaroslavsafar/Developer/HCN/models/belief_state_update/flan-t5-base-finetuned-2023-06-27-10-23-50/test_results.csv')
results_df = pd.read_csv(path, sep="\t")
results_df.dropna(inplace=True)

predictions_text = results_df['predicted_text'].tolist()
references_text = results_df['reference_text'].tolist()


# Convert the string representations back into dictionary format
predictions_dict = [str_to_belief_state(pred) for pred in predictions_text]
references_dict = [str_to_belief_state(ref) for ref in references_text]

"""
for p, r in zip(predictions_dict, references_dict):

    print(f"PRED = {p}, TYPE = {type(p)}")
    print(f"REFE = {r}, TYPE = {type(r)}")
    print(f"{p == r =}")

    print(f"=============================")
"""
metrics = compute_belief_state_metrics(references=references_dict, predictions=predictions_dict)
exact_match_ratio = compute_belief_state_exact_match_ratio(references=references_dict,
                                                           predictions=predictions_dict)
print(metrics)
metrics_dict = {
    "precision": metrics['global']['precision'],
    "recall": metrics['global']['recall'],
    "f1-score": metrics['global']['f1-score'],
    "exact_match_ratio": exact_match_ratio,
}

print(metrics_dict)

