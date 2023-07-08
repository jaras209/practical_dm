import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import datasets
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, pipelines, Pipeline, T5ForConditionalGeneration
from transformers.pipelines.base import KeyDataset

from database import MultiWOZDatabase
from metrics import compute_action_metrics, compute_actions_exact_match_ratio
from huggingface_multiwoz_dataset import MultiWOZDatasetActionGeneration, str_to_action_list, str_to_belief_state
from constants import DOMAIN_NAMES, OUTPUT_DF_COLUMNS
from accelerate import infer_auto_device_map

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

    results_df = results_df[['input_text', 'predicted_text', 'reference_text', 'predicted_list', 'reference_list']]

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


def create_dialogue_acts(actions: List[List[str]], belief_states: List[Dict[str, Dict[str, str]]],
                         database: MultiWOZDatabase) -> List[List[str]]:
    """
    Create dialogue acts for the actions by extending them with the corresponding output from the database query.

    Args:
        actions (List[List[str]]): A list of lists of actions.
        belief_states (List[Dict[str, Dict[str, str]]]): The list of belief states corresponding to the actions.
        database (MultiWOZDatabase): The database object with the query method.

    Returns:
        List[List[str]]: A list of lists of extended dialogue acts.
    """
    predicted_dialogue_acts = []
    for actions, belief_state in zip(actions, belief_states):

        dialogue_acts = []
        query_results_for_domain = {}
        for action in actions:
            if not action:
                logging.debug('Empty action')
                dialogue_acts.append('')
                continue

            match = re.match(r'(\w+)-(\w+)(?:\((\w+)\))?', action)
            if match:
                domain, action, slot = match.groups()
            else:
                raise ValueError(f"Action {action} doesn't match expected format")

            # Prepare base dialogue act
            dialogue_act_base = f"{domain}-{action}"
            dialogue_act = f"{dialogue_act_base}({slot})" if slot is not None else dialogue_act_base

            if domain in DOMAIN_NAMES:
                # Query the domain if it's not already queried
                if domain not in query_results_for_domain:
                    query_results_for_domain[domain] = database.query(domain, belief_state[domain])

                # If query result is not None and slot exists, modify the dialogue act with result from query.
                query_result = query_results_for_domain[domain]
                if query_result is not None and slot is not None:
                    for result in query_result:
                        if slot in result:
                            dialogue_act = f"{dialogue_act_base}({slot}={result[slot]})"
                            break

            dialogue_acts.append(dialogue_act)

        predicted_dialogue_acts.append(dialogue_acts)

    return predicted_dialogue_acts


def evaluate(dataset: MultiWOZDatasetActionGeneration, model_path: Path, only_dataset: str = None,
             max_target_length: int = 32,
             batch_size: int = 8) -> None:
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
                                             device=0 if device.type == 'cuda' else -1)

    logging.info(f"Pipeline created. Pipeline device: {classifier_pipeline.device}")

    start_time = time.time()

    # Evaluate the model on the datasets in multiwoz_dataset.
    for dataset_name, dataset_data in dataset.dataset.items():
        if only_dataset is not None and dataset_name != only_dataset:
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
        predictions_list = [str_to_action_list(pred) for pred in predictions_text]
        references_list = [str_to_action_list(ref) for ref in references_text]

        # Convert the belief states to dictionary format
        belief_states = [str_to_belief_state(belief_state_str) for belief_state_str in dataset_data['new_belief_state']]
        predicted_dialogue_acts = create_dialogue_acts(predictions_list, belief_states, dataset.database)
        reference_dialogue_acts = create_dialogue_acts(references_list, belief_states, dataset.database)

        # Create DataFrame with prediction results
        output_df = pd.DataFrame({
            'input_text': inputs_text,
            'predicted_text': predictions_text,
            'reference_text': references_text,
            'predicted_list': predictions_list,
            'reference_list': references_list,
            'predicted_dialogue_acts': predicted_dialogue_acts,
            'reference_dialogue_acts': reference_dialogue_acts
        })

        # Compute metrics
        logging.info("Computing metrics...")
        metrics = compute_action_metrics(references=references_list, predictions=predictions_list)

        # Save results
        logging.info("Saving results...")
        save_results(model_path, dataset_name, output_df, metrics)

        logging.info(f"Evaluating model on {dataset_name} done in {time.time() - dataset_start_time:.2f} seconds.\n")
        logging.info("=======================================================")

    logging.info(f"Evaluating model {model_path} on all datasets done in {time.time() - start_time:.2f} seconds.")
