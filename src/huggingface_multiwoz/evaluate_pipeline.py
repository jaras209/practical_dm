import argparse
from pathlib import Path
from typing import Union
import random
import logging
import json
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from transformers import pipelines

from database import MultiWOZDatabase
from evaluate_action_generation import create_dialogue_acts
from huggingface_multiwoz_dataset import belief_state_to_str, database_results_count_to_str, str_to_belief_state, \
    str_to_action_list, load_multiwoz_dataset
from metrics import compute_belief_state_metrics, compute_action_metrics
from constants import *


def save_dataframe(df: pd.DataFrame, save_path: Path, filename: str, also_json: bool = True):
    df.to_csv(save_path / f'{filename}.csv', index=True, sep='\t')
    if also_json:
        with open(save_path / f'{filename}.json', 'w') as f:
            json.dump(df.to_dict(orient='records'), f, indent=4)
        logging.info(f"Results saved to {save_path / f'{filename}.csv'} and {save_path / f'{filename}.json'}.")
    else:
        logging.info(f"Results saved to {save_path / f'{filename}.csv'}.")


def evaluate_models(df,
                    database: MultiWOZDatabase,
                    state_model_name_or_path: Union[str, Path] = None,
                    action_cla_model_name_or_path: Union[str, Path] = None,
                    action_gen_model_name_or_path: Union[str, Path] = None,
                    gen_tokenizer_name: str = 'google/flan-t5-base',
                    cla_tokenizer_name: str = 'roberta-base',
                    batch_size: int = 32,
                    max_source_length: int = 512,
                    max_target_length: int = 128,
                    use_predicted_states: bool = False,
                    save_path: Union[str, Path] = './',
                    dataset_name: str = 'results',
                    random_seed: int = 42):
    # Seed set for deterministic random subset selection
    np.random.seed(random_seed)
    random.seed(random_seed)
    logging.info("Starting models evaluation")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    generation_tokenizer = AutoTokenizer.from_pretrained(gen_tokenizer_name, use_fast=True)
    utterances = df['utterance']
    separator = CONTEXT_SEP
    contexts = list(map(lambda x: separator.join(x), df['context']))
    database_results_count = list(map(database_results_count_to_str, df['database_results']))

    # Load and create pipeline for dialogue state tracking model if provided
    if state_model_name_or_path is not None:
        logging.info(f"Loading dialogue state tracking model: {state_model_name_or_path}")
        state_model_name_or_path = Path(state_model_name_or_path)
        state_model = T5ForConditionalGeneration.from_pretrained(state_model_name_or_path).to(device)
        state_pipeline = pipelines.pipeline(task='text2text-generation',
                                            model=state_model,
                                            tokenizer=generation_tokenizer,
                                            device=0 if device.type == 'cuda' else -1)

        # Prepare inputs and references for the dialogue state tracking model
        old_states_texts = list(map(belief_state_to_str, df['old_belief_state']))
        state_input_texts = list(map(lambda belief, context, user_utter:
                                     TASK_DESCRIPTION_STATE_UPDATE + ' ' + STATE_GEN + ' ' + belief + '. '
                                     + CONTEXT_GEN + ' ' + context + '. ' + USER_GEN + ' ' + user_utter,
                                     old_states_texts, contexts, utterances))
        # Make predictions with the dialogue state tracking model
        logging.info("Computing predictions for dialogue state tracking model")
        state_predictions = state_pipeline(state_input_texts, max_length=max_target_length, batch_size=batch_size)
        state_predicted_texts = [pred['generated_text'] for pred in state_predictions]
        state_reference_texts = list(map(belief_state_to_str, df['new_belief_state']))

        # Make dictionaries from the predictions and references (references also to remove None values, in df dataset
        # we have None values in dictionaries included to have fixed structure required by huggingface datasets)
        state_predicted_dicts = [str_to_belief_state(pred) for pred in state_predicted_texts]
        state_reference_dicts = [str_to_belief_state(pred) for pred in state_reference_texts]

        # Compute metrics
        logging.info("Computing metrics for dialogue state tracking model")
        state_metrics = compute_belief_state_metrics(references=state_reference_dicts,
                                                     predictions=state_predicted_dicts)
        state_metrics_df = pd.DataFrame.from_dict(state_metrics, orient='index')
        save_dataframe(state_metrics_df, save_path, f'{dataset_name}_state_metrics.csv', False)
        logging.info("Dialogue state tracking model metrics saved")
        logging.info("Dialogue state tracking model evaluation finished\n\n")
    else:
        logging.info("Dialogue state tracking model not provided")
        # If the dialogue state tracking model is not provided, use the ground truth new belief states
        state_predicted_texts = list(map(belief_state_to_str, df['new_belief_state']))

    # Load and create pipeline for action generation model if provided
    if action_gen_model_name_or_path is not None:
        logging.info(f"Loading action generation model: {action_gen_model_name_or_path}")
        action_gen_model = T5ForConditionalGeneration.from_pretrained(action_gen_model_name_or_path).to(device)
        action_gen_pipeline = pipelines.pipeline(task='text2text-generation',
                                                 model=action_gen_model,
                                                 tokenizer=generation_tokenizer,
                                                 device=0 if device.type == 'cuda' else -1)

        # Prepare inputs for the action generation model

        new_belief_states_texts = state_predicted_texts if use_predicted_states else list(
            map(belief_state_to_str, df['new_belief_state']))
        new_belief_states_dicts = [str_to_belief_state(s) for s in new_belief_states_texts]
        action_gen_input_texts = list(map(lambda belief, context, user_utter, db_results_count:
                                          TASK_DESCRIPTION_ACTION_GENERATION + ' ' + STATE_GEN + ' ' + belief + '. '
                                          + CONTEXT_GEN + ' ' + context + '. ' + USER_GEN + ' ' + user_utter + '. ' +
                                          DATABASE_COUNTS_GEN + ' ' + db_results_count,
                                          new_belief_states_texts, contexts, utterances, database_results_count))

        # Make predictions with the action generation model
        logging.info("Computing predictions for action generation model")
        action_gen_predictions = action_gen_pipeline(action_gen_input_texts, max_length=max_target_length,
                                                     batch_size=batch_size)
        action_gen_predicted_texts = [pred['generated_text'] for pred in action_gen_predictions]

        action_gen_predicted_lists = [str_to_action_list(pred) for pred in action_gen_predicted_texts]
        action_gen_reference_lists = df['actions'].tolist()

        # Compute metrics
        logging.info("Computing metrics for action generation model")
        action_gen_metrics = compute_action_metrics(references=action_gen_reference_lists,
                                                    predictions=action_gen_predicted_lists)
        action_gen_metrics_df = pd.DataFrame.from_dict(action_gen_metrics, orient='index')
        save_dataframe(action_gen_metrics_df, save_path, f'{dataset_name}_action_gen_metrics', False)
        logging.info("Action generation model metrics saved")

        # Create dialogue acts
        logging.info("Computing dialogue acts for action generation model")
        action_gen_predicted_da = create_dialogue_acts(action_gen_predicted_lists, new_belief_states_dicts, database)

        logging.info("Action generation model evaluation finished\n\n")
    # Load and create pipeline for action classification model if provided
    if action_cla_model_name_or_path is not None:
        logging.info(f"Loading action classification model: {action_cla_model_name_or_path}")
        action_cla_model = AutoModelForSequenceClassification.from_pretrained(
            action_cla_model_name_or_path).to(device)
        action_cla_tokenizer = AutoTokenizer.from_pretrained(cla_tokenizer_name, use_fast=True)
        action_cla_pipeline = pipelines.pipeline('text-classification',
                                                 model=action_cla_model,
                                                 tokenizer=action_cla_tokenizer,
                                                 top_k=5,
                                                 device=0 if device.type == 'cuda' else -1)

        new_belief_states_texts = state_predicted_texts if use_predicted_states else list(
            map(belief_state_to_str, df['new_belief_state']))
        new_belief_states_dicts = [str_to_belief_state(s) for s in new_belief_states_texts]

        # Prepare inputs for the action classification model
        action_cla_input_texts = list(map(lambda belief, context, user_utter, db_results_count:
                                          STATE_CLA + ' ' + belief + '. '
                                          + CONTEXT_CLA + ' ' + context + '. ' + USER + ' ' + user_utter + '. ' +
                                          DATABASE_COUNTS + ' ' + db_results_count,
                                          new_belief_states_texts, contexts, utterances, database_results_count))

        # Make predictions with the action classification model
        logging.info("Computing predictions for action classification model")
        action_cla_predictions = action_cla_pipeline(action_cla_input_texts, batch_size=batch_size)
        action_cla_predicted_lists = [[p['label'] for p in pred if p['score'] >= 0.5] for pred in action_cla_predictions]
        action_cla_reference_lists = df['actions'].tolist()

        # Compute metrics
        logging.info("Computing metrics for action classification model")
        action_cla_metrics = compute_action_metrics(references=action_cla_reference_lists,
                                                    predictions=action_cla_predicted_lists)
        action_cla_metrics_df = pd.DataFrame.from_dict(action_cla_metrics, orient='index')
        save_dataframe(action_cla_metrics_df, save_path, f'{dataset_name}_action_cla_metrics', False)
        logging.info("Action classification model metrics saved")

        # Create dialogue acts
        logging.info("Computing dialogue acts for action classification model")
        action_cla_predicted_da = create_dialogue_acts(action_cla_predicted_lists, new_belief_states_dicts, database)
        logging.info("Action classification model evaluation finished\n\n")

    # Prepare output data frame
    results_df = df.copy()
    if state_model_name_or_path is not None:
        results_df['state_input'] = state_input_texts
        results_df['state_refe_text'] = state_reference_texts
        results_df['state_pred_text'] = state_predicted_texts
        results_df['state_refe_dict'] = state_reference_dicts
        results_df['state_pred_dict'] = state_predicted_dicts

    if action_gen_model_name_or_path is not None:
        results_df['action_input'] = action_gen_input_texts
        results_df['action_refe'] = action_gen_reference_lists
        results_df['action_gen_pred_list'] = action_gen_predicted_lists
        results_df['action_gen_pred_da'] = action_gen_predicted_da

    if action_cla_model_name_or_path is not None:
        if action_gen_model_name_or_path is None:
            results_df['action_input'] = action_cla_input_texts
            results_df['action_refe'] = action_cla_reference_lists

        results_df['action_cla_pred_list'] = action_cla_predicted_lists
        results_df['action_cla_pred_da'] = action_cla_predicted_da

    # Reorder columns
    results_df = results_df[
        ['dialogue_id', 'user_turn_id', 'system_turn_id', 'utterance', 'context', 'system_utterance',
         'state_input', 'state_refe_text', 'state_pred_text', 'state_refe_dict', 'state_pred_dict',
         'action_input', 'action_refe', 'action_gen_pred_list', 'action_cla_pred_list', 'action_gen_pred_da',
         'action_cla_pred_da']]

    # Save full results
    save_dataframe(results_df, save_path, f'{dataset_name}_results')

    # Save a subset of results (random subset of 30 dialogues)
    random.seed(42)
    unique_dialogue_ids = results_df['dialogue_id'].unique()
    subset_dialogue_ids = random.sample(list(unique_dialogue_ids), min(30, len(unique_dialogue_ids)))
    results_df_subset = results_df[results_df['dialogue_id'].isin(subset_dialogue_ids)]

    save_dataframe(results_df_subset, save_path, f'{dataset_name}_results_subset')

    # Create state tracking output data frame
    if state_model_name_or_path is not None:
        state_tracking_df = results_df[
            ['dialogue_id', 'user_turn_id', 'system_turn_id', 'state_input', 'state_refe_text', 'state_pred_text',
             'state_refe_dict', 'state_pred_dict']]
        save_dataframe(state_tracking_df, save_path, f'{dataset_name}_state_tracking_results')

    # Create action selection output data frame
    if action_gen_model_name_or_path is not None or action_cla_model_name_or_path is not None:
        action_selection_df = results_df[
            ['dialogue_id', 'user_turn_id', 'system_turn_id', 'action_input', 'action_refe', 'action_gen_pred_list',
             'action_cla_pred_list', 'action_gen_pred_da', 'action_cla_pred_da']]
        save_dataframe(action_selection_df, save_path, f'{dataset_name}_action_selection_results')

    logging.info("Evaluation completed")


def main():
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('--state_model_name_or_path', type=str,
                        default="../../models/belief_state_update/flan-t5-base-finetuned-2023-07-18-07-06-36",
                        help='Path or name of the state model')
    parser.add_argument('--action_cla_model_name_or_path', type=str,
                        default="../../models/models/action_classification/roberta-base-finetuned-2023-07-17-18-36-51",
                        help='Path or name of the action classification model')
    parser.add_argument('--action_gen_model_name_or_path', type=str,
                        default="../../models/models/action_generation/flan-t5-base-finetuned-2023-07-18-21-00-58",
                        help='Path or name of the action generation model')
    parser.add_argument('--gen_tokenizer_name', type=str, default='google/flan-t5-base',
                        help='Name of the generation tokenizer')
    parser.add_argument('--cla_tokenizer_name', type=str, default='roberta-base',
                        help='Name of the classification tokenizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_source_length', type=int, default=260, help='Maximum source length')
    parser.add_argument('--max_target_length', type=int, default=230, help='Maximum target length')
    parser.add_argument('--use_predicted_states', type=bool, default=False,
                        help='Whether to use predicted belief states')
    parser.add_argument('--save_path', type=str, default="../../results/test100/bla", help='Path to save the results')
    parser.add_argument('--dataset_name', type=str, default='test', help='Name of the dataset')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument("--data_path",
                        default="../../data/huggingface_data",
                        # default="../../data/huggingface_data",
                        type=str,
                        help="Name of the folder where to save extracted multiwoz dataset for faster preprocessing.")

    args = parser.parse_args()

    # Load MultiWoz Database, which is locally saved at database_path
    database_path = Path(args.data_path) / "database"
    database = MultiWOZDatabase(database_path)

    # Load your DataFrame here (df)
    test_df = load_multiwoz_dataset('test', database=database, root_cache_path=args.data_path)
    print(test_df[test_df['utterance'] == "i need a place to dine in the center thats expensive"]['dialogue_id'])

    evaluate_models(test_df,
                    database=database,
                    state_model_name_or_path=args.state_model_name_or_path,
                    action_cla_model_name_or_path=args.action_cla_model_name_or_path,
                    action_gen_model_name_or_path=args.action_gen_model_name_or_path,
                    gen_tokenizer_name=args.gen_tokenizer_name,
                    cla_tokenizer_name=args.cla_tokenizer_name,
                    batch_size=args.batch_size,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                    use_predicted_states=args.use_predicted_states,
                    save_path=args.save_path,
                    dataset_name=args.dataset_name,
                    random_seed=args.random_seed)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
