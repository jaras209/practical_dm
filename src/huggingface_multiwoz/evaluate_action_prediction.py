import json
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
from transformers import AutoModelForSequenceClassification, pipelines, Pipeline
from transformers.pipelines.base import KeyDataset

from database import MultiWOZDatabase
from huggingface_multiwoz_dataset import MultiWOZDatasetActions
from constants import DOMAIN_NAMES, OUTPUT_DF_COLUMNS

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(level=logging.INFO)


def create_dialogue_acts(predicted_labels: List[List[str]], dataset: datasets.Dataset, database: MultiWOZDatabase):
    """
    Create dialogue acts for the predicted labels by extending them with the corresponding output from the database query.

    Args:
        predicted_labels (List[List[str]]): A list of lists of predicted labels.
        dataset (datasets.Dataset): The dataset containing the input dialogue elements.
        database (MultiWOZDatabase): The database object with the query method.

    Returns:
        List[List[str]]: A list of lists of extended dialogue acts.
    """
    predicted_dialogue_acts = []
    for i, (labels, dataset_element) in enumerate(zip(predicted_labels, dataset)):
        belief_state = dataset_element['new_belief_state']

        dialogue_acts = []
        for label in labels:
            if not label:
                logging.debug(f'Empty label at index {i}')
                dialogue_acts.append('')
                continue

            parts = label.split('-')

            if len(parts) > 2:
                domain, action, slot = parts[-3], parts[-2], parts[-1]
            elif len(parts) == 2:
                domain, action = parts[-2], parts[-1]
                slot = None
            else:
                domain, action = parts[-1], None
                slot = None

            domain = domain.lower()

            # Prepare base dialogue act
            dialogue_act_base = f"{domain}-{action}"
            dialogue_act = f"{dialogue_act_base}({slot})" if slot is not None else dialogue_act_base

            if domain in DOMAIN_NAMES:
                # Query the domain to get the full dialogue act.
                query_result = database.query(domain, belief_state[domain])

                # If query result is not None and slot exists, modify the dialogue act with result from query.
                if query_result is not None and slot is not None:
                    for result in query_result:
                        if slot in result:
                            dialogue_act = f"{dialogue_act_base}({slot}={result[slot]})"
                            break

            dialogue_acts.append(dialogue_act)

        predicted_dialogue_acts.append(dialogue_acts)

    return predicted_dialogue_acts


def get_predictions(classifier_pipeline: Pipeline, dataset_data: datasets.Dataset) -> List[List[Dict[str, Any]]]:
    """
    Get model predictions for the dataset.

    Args:
        classifier_pipeline (Pipeline): Hugging Face pipeline for text classification.
        dataset_data (Dataset): Hugging Face Dataset object containing the dataset data.

    Returns:
        List[List[Dict[str, Any]]]: List of predictions, where each prediction is a list of dicts
                                    with keys 'label' and 'score' for each dialogue.
    """
    # Prepare the key dataset for the model pipeline.
    key_dataset = KeyDataset(dataset_data, 'text')

    # Compute predictions using the model pipeline.
    predictions = classifier_pipeline(key_dataset)

    return predictions


def compare_labels(predictions: List[List[Dict[str, Any]]], dataset_data: datasets.Dataset, label2id: Dict[str, int]) -> \
        tuple[list[list[Any]], ndarray, ndarray]:
    """
    Compare the predicted labels with the true labels.

    Args:
        predictions (List[List[Dict[str, Any]]]): List of predictions, where each prediction is a list of dicts
                                                 with keys 'label' and 'score' for each dialogue.
        dataset_data (Dataset): Hugging Face Dataset object containing the dataset data.
        label2id (Dict[str, int]): Dictionary mapping labels to their corresponding IDs.

    Returns:
        Tuple[np.ndarray, List[List[str]]]: Tuple containing the binary array of predicted labels and
                                           the list of predicted label strings.
    """
    predicted_labels = []
    y_pred = np.zeros((len(predictions), len(label2id)), dtype='bool')
    y_true = np.array(dataset_data['label'])

    for i, prediction in enumerate(predictions):
        # i-th `prediction` is a model output for the i-th input dialogue.
        # It is a list of dict items with the following format:
        #   - len: number of predicted actions - 'top_k' in the pipeline
        #   - elements: dict with the following key-value pairs:
        #       - prediction['label']: action name
        #       - prediction['score']: probability of this action
        labels = []
        labels_ids = []
        for pred in prediction:
            score = round(pred['score'], 4)
            action = pred['label']
            if score >= 0.5:
                labels.append(action)
                labels_ids.append(label2id[action])

        predicted_labels.append(labels)
        y_pred[i, labels_ids] = 1

    return predicted_labels, y_pred, y_true


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, id2label: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate evaluation metrics (recall, precision, f1-score, and accuracy) for the given true and predicted labels.

    Args:
        y_true (np.ndarray): Binary array of true labels.
        y_pred (np.ndarray): Binary array of predicted labels.
        id2label (Dict[int, str]): Dictionary mapping label IDs to their corresponding labels.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing evaluation metrics and their respective values.
    """
    # Calculate recall scores
    actions_recall = recall_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
    macro_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    weighted_recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    recall = {'metric': 'recall', 'macro': round(macro_recall, 4), 'weighted': round(weighted_recall, 4)}
    recall.update({action: round(actions_recall[i], 4) for i, action in id2label.items()})

    # Calculate precision scores
    actions_precision = precision_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
    macro_precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    weighted_precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    precision = {'metric': 'precision', 'macro': round(macro_precision, 4), 'weighted': round(weighted_precision, 4)}
    precision.update({action: round(actions_precision[i], 4) for i, action in id2label.items()})

    # Calculate F1 scores
    actions_f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    f1 = {'metric': 'f1', 'macro': round(macro_f1, 4), 'weighted': round(weighted_f1, 4)}
    f1.update({action: round(actions_f1[i], 4) for i, action in id2label.items()})

    # Calculate accuracy
    accuracy = {'metric': 'accuracy', 'macro': round(accuracy_score(y_true=y_true, y_pred=y_pred), 4)}

    return {'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': accuracy}


def save_results(model_path: Path, dataset_name: str, output_df: pd.DataFrame,
                 metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Save the prediction results and evaluation metrics to CSV files.

    Args:
        model_path (Path): The path to the directory where the model is stored.
        dataset_name (str): The name of the dataset being evaluated.
        output_df (pd.DataFrame): The DataFrame containing the prediction results.
        metrics (Dict[str, Dict[str, float]]): The dictionary containing the evaluation metrics.
    """
    # Save predictions to file.
    output_df = output_df[OUTPUT_DF_COLUMNS].transpose()
    model_path.mkdir(exist_ok=True, parents=True)

    output_df.to_csv(model_path / f'{dataset_name}_results.csv', index=False, sep='\t')
    with open(model_path / f'{dataset_name}_results.json', 'w') as f:
        json.dump(output_df.to_dict(orient='records'), f, indent=4)

    logging.info(f"Results saved to {model_path / f'{dataset_name}_results.csv'} and "
                 f"{model_path / f'{dataset_name}_results.json'}.")

    # Convert metrics dictionary to DataFrame and save to file.
    metrics_df = pd.DataFrame([metrics['accuracy'], metrics['recall'], metrics['precision'], metrics['f1']])
    logging.info(metrics_df)
    metrics_df.to_csv(model_path / f'{dataset_name}_metrics.csv')
    logging.info(f"Metrics saved to {model_path / f'{dataset_name}_metrics.csv'}.")


def evaluate(multiwoz_dataset: MultiWOZDatasetActions, model_path: Path, top_k: int = 5,
             only_dataset: str = None):
    """
    Evaluate the model on the datasets in multiwoz_dataset.

    Args:
        multiwoz_dataset (MultiWOZDatasetActions): The dataset object containing the dataset and tokenizer.
        model_path (Path): The path to the trained model.
        top_k (int, optional): The top k actions to consider in the model pipeline. Defaults to 5.
        only_dataset (str, optional): If set, only evaluate the model on the specified dataset. Defaults to None.
    """
    if only_dataset is not None and only_dataset not in multiwoz_dataset.dataset:
        raise ValueError(f"Dataset {only_dataset} not found in multiwoz_dataset.")

    logging.info(f"Evaluating model {model_path}...")

    # Load the trained model.
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # Prepare model for evaluation.
    model.eval()

    # Create the pipeline to use for inference.
    classifier_pipeline = pipelines.pipeline(task='text-classification',
                                             model=model,
                                             tokenizer=multiwoz_dataset.tokenizer,
                                             top_k=top_k,
                                             device=0 if torch.cuda.is_available() else -1)

    start_time = time.time()

    # Evaluate the model on the datasets in multiwoz_dataset.
    for dataset_name, dataset_data in multiwoz_dataset.dataset.items():
        if only_dataset is not None and dataset_name != only_dataset:
            continue

        logging.info(f"Evaluating model on {dataset_name}...")
        dataset_start_time = time.time()

        # Compute predictions using the model pipeline.
        predictions_start_time = time.time()
        predictions = get_predictions(classifier_pipeline, dataset_data)
        logging.info(f"Predictions computed in {time.time() - predictions_start_time:.2f} seconds.")

        # Compare labels
        compare_start_time = time.time()
        predicted_labels, y_pred, y_true = compare_labels(predictions, dataset_data, label2id=multiwoz_dataset.get_label2id())
        logging.info(f"Comparing predictions with true labels took {time.time() - compare_start_time:.2f} seconds.")

        # Create dialogue acts from the predicted labels.
        dialogue_acts_start_time = time.time()
        dialogue_acts = create_dialogue_acts(predicted_labels, dataset_data, multiwoz_dataset.database)
        logging.info(f"Creating dialogue acts took {time.time() - dialogue_acts_start_time:.2f} seconds.")

        # Calculate metrics
        metrics_start_time = time.time()
        metrics = calculate_metrics(y_true, y_pred, id2label=multiwoz_dataset.get_id2label())
        logging.info(f"Computing metrics for {dataset_name} took {time.time() - metrics_start_time:.2f} seconds.")

        # Create output_df containing prediction results
        output_df = dataset_data.to_pandas()
        output_df['predicted'] = predicted_labels
        output_df['dialogue_acts'] = dialogue_acts
        output_df['scores'] = predictions

        # Save results
        save_results(model_path, dataset_name, output_df, metrics)

        logging.info(f"Evaluating model on {dataset_name} done in {time.time() - dataset_start_time:.2f} seconds.\n")
        logging.info("=======================================================")

    logging.info(f"Evaluating model {model_path} on all datasets done in {time.time() - start_time:.2f} seconds.")
