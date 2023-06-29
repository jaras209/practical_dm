import json
from pathlib import Path

import numpy as np
import pandas as pd

from huggingface_multiwoz_dataset import str_to_belief_state

path = Path('/Users/jaroslavsafar/Developer/HCN/models/belief_state_update/flan-t5-base-finetuned-2023-06-27-10-23-50')
dataset_name = 'test'
results_df = pd.read_csv(path / f'{dataset_name}_results.csv')[['input_text', 'predicted_text', 'reference_text']]

# Replace NaN values with an empty string
results_df = results_df.replace(np.nan, '')

predictions_text = results_df['predicted_text'].tolist()
references_text = results_df['reference_text'].tolist()

# Convert the string representations back into dictionary format
predictions_dict = [str_to_belief_state(pred) for pred in predictions_text]
references_dict = [str_to_belief_state(ref) for ref in references_text]

# Add predictions and references in dictionary format to the DataFrame
results_df['predicted_dict'] = predictions_dict
results_df['reference_dict'] = references_dict



with open(path / f'{dataset_name}_results.json', 'w') as f:
    json.dump(results_df.to_dict(orient='records'), f, indent=4)

results_df_subset = results_df.head(min(1000, len(results_df)))
results_df_subset.to_csv(path / f'{dataset_name}_results_subset.csv', index=False, sep='\t')

with open(path / f'{dataset_name}_results_subset.json', 'w') as f:
    json.dump(results_df_subset.to_dict(orient='records'), f, indent=4)
