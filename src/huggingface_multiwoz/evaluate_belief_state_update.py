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
from huggingface_multiwoz_dataset import MultiWOZBeliefUpdate
from constants import DOMAIN_NAMES, OUTPUT_DF_COLUMNS

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(level=logging.INFO)


def get_predictions(classifier_pipeline: Pipeline, input_text: List[str]) -> List[List[Dict[str, Any]]]:
    """
    Get model predictions for the dataset.

    Args:
        classifier_pipeline (Pipeline): Hugging Face pipeline for text classification.
        input_text (Dataset): Input text for the model pipeline.

    Returns:
        List[List[Dict[str, Any]]]: List of predictions, where each prediction is a list of dicts
                                    with keys 'label' and 'score' for each dialogue.
    """
    # Compute predictions using the model pipeline.
    predictions = classifier_pipeline(input_text)

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
    output_df = output_df[OUTPUT_DF_COLUMNS]
    model_path.mkdir(exist_ok=True, parents=True)
    output_df.to_csv(model_path / f'{dataset_name}_predictions.csv')
    logging.info(f"Predictions saved to {model_path / f'{dataset_name}_predictions.csv'}.")

    # Convert metrics dictionary to DataFrame and save to file.
    metrics_df = pd.DataFrame([metrics['accuracy'], metrics['recall'], metrics['precision'], metrics['f1']])
    logging.info(metrics_df)
    metrics_df.to_csv(model_path / f'{dataset_name}_metrics.csv')
    logging.info(f"Metrics saved to {model_path / f'{dataset_name}_metrics.csv'}.")


def evaluate(dataset: MultiWOZBeliefUpdate, model_path: Path, only_dataset: str = None):
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
        input_ids = KeyDataset(dataset_data, 'input_ids')
        label_ids = np.array(dataset_data['labels'])
        label_ids[label_ids == -100] = dataset.tokenizer.pad_token_id

        input_text = dataset.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[:50]
        label_text = dataset.tokenizer.batch_decode(label_ids, skip_special_tokens=True)[:50]

        # Compute predictions using the model pipeline.
        predictions = get_predictions(classifier_pipeline, input_text)

        for input_, output_, label_ in zip(input_text, predictions, label_text):
            print(f'Input: {input_}')
            print(f'Output: {output_["generated_text"]}')
            print(f'Label: {label_}')
            print(f"{'-' * 50}")

        exit()

        # Compare labels
        predicted_labels, y_pred, y_true = compare_labels(predictions, dataset_data, label2id=dataset.get_label2id())

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, id2label=dataset.get_id2label())

        # Create output_df containing prediction results
        output_df = dataset_data.to_pandas()
        output_df['predicted_belief_state'] = predictions
        output_df['scores'] = predictions

        # Save results
        save_results(model_path, dataset_name, output_df, metrics)

        logging.info(f"Evaluating model on {dataset_name} done in {time.time() - dataset_start_time:.2f} seconds.\n")
        logging.info("=======================================================")

    logging.info(f"Evaluating model {model_path} on all datasets done in {time.time() - start_time:.2f} seconds.")
