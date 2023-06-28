import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import datasets
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, pipelines, Pipeline, T5ForConditionalGeneration
from transformers.pipelines.base import KeyDataset

from database import MultiWOZDatabase
from metrics import compute_belief_state_metrics, compute_belief_state_exact_match_ratio
from huggingface_multiwoz_dataset import MultiWOZBeliefUpdate, str_to_belief_state
from constants import DOMAIN_NAMES, OUTPUT_DF_COLUMNS

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
    results_df = results_df[['input_text', 'predicted_text', 'reference_text']]

    # Save results to CSV
    results_df.to_csv(model_path / f'{dataset_name}_results.csv')
    logging.info(f"Results saved to {model_path / f'{dataset_name}_results.csv'}.")

    # Save results to JSON
    results_df.to_json(model_path / f'{dataset_name}_results.json', orient='records', lines=True)
    logging.info(f"Results saved to {model_path / f'{dataset_name}_results.json'}.")

    # Convert metrics dictionary to DataFrame and save to file.
    metrics_df = pd.DataFrame(metrics).transpose()

    # Save metrics to CSV
    metrics_df.to_csv(model_path / f'{dataset_name}_metrics.csv')
    logging.info(f"Metrics saved to {model_path / f'{dataset_name}_metrics.csv'}.")

    # Save metrics to JSON
    metrics_df.to_json(model_path / f'{dataset_name}_metrics.json', orient='columns')
    logging.info(f"Metrics saved to {model_path / f'{dataset_name}_metrics.json'}.")


def evaluate(dataset: MultiWOZBeliefUpdate, model_path: Path, only_dataset: str = None, max_target_length: int = 32):
    """
    Evaluate the model on the datasets in multiwoz_dataset.

    Args:
        dataset (MultiWOZDatasetActions): The dataset object containing the dataset and tokenizer.
        model_path (Path): The path to the trained model.
        top_k (int, optional): The top k actions to consider in the model pipeline. Defaults to 5.
        only_dataset (str, optional): If set, only evaluate the model on the specified dataset. Defaults to None.
    """
    if only_dataset is not None and only_dataset not in dataset.dataset:
        raise ValueError(f"Dataset {only_dataset} not found in multiwoz_dataset.")

    logging.info(f"Evaluating model {model_path}...")

    # Load the trained model.
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    # Prepare model for evaluation.
    model.eval()

    # Create the pipeline to use for inference.
    classifier_pipeline = pipelines.pipeline(task='text2text-generation',
                                             model=model,
                                             tokenizer=dataset.tokenizer,
                                             device=0 if torch.cuda.is_available() else -1)

    start_time = time.time()

    # Evaluate the model on the datasets in multiwoz_dataset.
    for dataset_name, dataset_data in dataset.dataset.items():
        if only_dataset is not None and dataset_name != only_dataset:
            continue

        dataset_start_time = time.time()
        logging.info(f"Evaluating model on {dataset_name}...")

        # Prepare the key dataset for the model pipeline and get the input text.
        input_ids = np.array(dataset_data['input_ids'])
        label_ids = np.array(dataset_data['labels'])

        # Assign pad_token_id back to those label ids with value -100, which is used for skipping loss calculation
        # on them
        label_ids[label_ids == -100] = dataset.tokenizer.pad_token_id

        # Convert input and label ids to text
        inputs_text = dataset.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        references_text = dataset.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute predictions using the model pipeline
        predictions = classifier_pipeline(inputs_text, max_length=max_target_length)
        predictions_text = [pred['generated_text'] for pred in predictions]

        # Convert the string representations back into dictionary format
        predictions_dict = [str_to_belief_state(pred) for pred in predictions_text]
        references_dict = [str_to_belief_state(ref) for ref in references_text]

        # Create DataFrame with prediction results
        output_df = pd.DataFrame({
            'input_text': inputs_text,
            'predicted_text': predictions_text,
            'reference_text': references_text,
        })

        # Compute metrics
        metrics = compute_belief_state_metrics(references_dict, predictions_dict)

        # Save results
        save_results(model_path, dataset_name, output_df, metrics)

        logging.info(f"Evaluating model on {dataset_name} done in {time.time() - dataset_start_time:.2f} seconds.\n")
        logging.info("=======================================================")

    logging.info(f"Evaluating model {model_path} on all datasets done in {time.time() - start_time:.2f} seconds.")
