import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import torch

from transformers import pipelines, T5ForConditionalGeneration

from metrics import compute_belief_state_metrics
from huggingface_multiwoz_dataset import MultiWOZDatasetBeliefUpdate, str_to_belief_state

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(level=logging.INFO)


def save_results(model_path: Path, dataset_name: str, results_df: pd.DataFrame,
                 metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Save the prediction results and evaluation metrics to CSV and JSON files.

    Args:
        model_path (Path): The path to the directory where the model is stored.
        dataset_name (str): The name of the dataset being evaluated.
        results_df (pd.DataFrame): The DataFrame containing the prediction results.
        metrics (Dict[str, Dict[str, float]]): The dictionary containing the evaluation metrics.
    """
    # Save results to file.
    model_path.mkdir(exist_ok=True, parents=True)

    results_df = results_df[['input_text', 'predicted_text', 'reference_text', 'predicted_dict', 'reference_dict']]

    # Replace NaN values with an empty string
    results_df = results_df.replace(np.nan, '')

    # Save full results
    results_df.to_csv(model_path / f'{dataset_name}_results.csv', index=False, sep='\t')
    with open(model_path / f'{dataset_name}_results.json', 'w') as f:
        json.dump(results_df.to_dict(orient='records'), f, indent=4)

    logging.info(f"Full results saved to {model_path / f'{dataset_name}_results.csv'} and "
                 f"{model_path / f'{dataset_name}_results.json'}.")

    # Save a subset of results
    results_df_subset = results_df.head(min(1000, len(results_df)))
    results_df_subset.to_csv(model_path / f'{dataset_name}_results_subset.csv', index=False, sep='\t')

    with open(model_path / f'{dataset_name}_results_subset.json', 'w') as f:
        json.dump(results_df_subset.to_dict(orient='records'), f, indent=4)

    logging.info(f"Subset of results saved to {model_path / f'{dataset_name}_results_subset.csv'} and "
                 f"{model_path / f'{dataset_name}_results_subset.json'}.")

    # Convert metrics dictionary to DataFrame and save to file.
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.to_csv(model_path / f'{dataset_name}_metrics.csv', index=True, sep='\t')
    logging.info(f"Metrics saved to {model_path / f'{dataset_name}_metrics.csv'}.")


def evaluate(dataset: MultiWOZDatasetBeliefUpdate, model_path: Path, only_dataset: Union[str, List[str]] = None,
             max_target_length: int = 32,
             batch_size: int = 16) -> None:
    """
    Evaluate the model on the datasets in multiwoz_dataset.

    Args:
        dataset (MultiWOZDatasetActions): The dataset object containing the dataset and tokenizer.
        model_path (Path): The path to the trained model.
        top_k (int, optional): The top k actions to consider in the model pipeline. Defaults to 5.
        only_dataset (str, optional): If set, only evaluate the model on the specified datasets. Defaults to None.
    """
    dataset_names = []
    if only_dataset is not None:
        if isinstance(only_dataset, str):
            if only_dataset not in dataset.dataset:
                raise ValueError(f"Dataset {only_dataset} not found in multiwoz_dataset.")
            else:
                dataset_names.append(only_dataset)

        elif isinstance(only_dataset, list):
            for d in only_dataset:
                if isinstance(d, str):
                    if d not in dataset.dataset:
                        raise ValueError(f"Dataset {only_dataset} not found in multiwoz_dataset.")

                    else:
                        dataset_names.append(d)

                else:
                    raise ValueError(f"Dataset {only_dataset} is not string.")
        else:
            raise ValueError(f"{only_dataset} is not string nor list of strings.")

    else:
        dataset_names = list[dataset.dataset.keys()]

    logging.info(f"Evaluating model {model_path}...")

    # Load the trained model.
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    # Prepare model for evaluation.
    model.eval()

    # Create the pipeline to use for inference.
    classifier_pipeline = pipelines.pipeline(task='text2text-generation',
                                             model=model,
                                             tokenizer=dataset.tokenizer,
                                             device=0 if device.type == 'cuda' else -1)

    logging.info(f"Pipeline created. Pipeline device: {classifier_pipeline.device}")

    start_time = time.time()

    # Evaluate the model on the datasets in multiwoz_dataset.
    for dataset_name, dataset_data in dataset.dataset.items():
        if dataset_name not in dataset_names:
            continue

        dataset_start_time = time.time()
        logging.info(f"Evaluating model on {dataset_name}...")

        # Prepare the key dataset for the model pipeline and get the input text.
        input_ids = torch.tensor(dataset_data['input_ids'])
        label_ids = torch.tensor(dataset_data['labels'])

        # Assign pad_token_id back to those label ids with value -100, which is used for skipping loss calculation
        # on them
        label_ids[label_ids == -100] = dataset.tokenizer.pad_token_id

        # Convert input and label ids to text
        logging.info("Converting input and label ids to text...")
        inputs_text = dataset.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        references_text = dataset.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute predictions using the model pipeline
        logging.info("Computing predictions...")
        predictions = classifier_pipeline(inputs_text, max_length=max_target_length, batch_size=batch_size)
        predictions_text = [pred['generated_text'] for pred in predictions]

        # Convert the string representations back into dictionary format
        logging.info("Converting predictions and references to dictionary format...")
        predictions_dict = [str_to_belief_state(pred) for pred in predictions_text]
        references_dict = [str_to_belief_state(ref) for ref in references_text]

        # Create DataFrame with prediction results
        output_df = pd.DataFrame({
            'input_text': inputs_text,
            'predicted_text': predictions_text,
            'reference_text': references_text,
            'predicted_dict': predictions_dict,
            'reference_dict': references_dict
        })

        # Compute metrics
        logging.info("Computing metrics...")
        metrics = compute_belief_state_metrics(references_dict, predictions_dict)

        # Save results
        logging.info("Saving results...")
        save_results(model_path, dataset_name, output_df, metrics)

        logging.info(f"Evaluating model on {dataset_name} done in {time.time() - dataset_start_time:.2f} seconds.\n")
        logging.info("=======================================================")

    logging.info(f"Evaluating model {model_path} on all datasets done in {time.time() - start_time:.2f} seconds.")
