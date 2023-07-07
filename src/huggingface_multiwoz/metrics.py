from collections import defaultdict
from pathlib import Path
import csv
from typing import Tuple, List, Dict

import numpy as np
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, EvalPrediction
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


import evaluate

from constants import DOMAIN_SLOTS
from huggingface_multiwoz_dataset import str_to_belief_state, str_to_action_list


class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if 'metrics' in kwargs:
            output_eval_file = Path(args.output_dir) / "eval_metrics.csv"
            is_first_write = not output_eval_file.is_file()
            with open(output_eval_file, 'a', newline='') as writer:
                fieldnames = ['epoch', 'global_step'] + list(kwargs['metrics'].keys())
                csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
                if is_first_write:
                    csv_writer.writeheader()
                row_dict = {'epoch': state.epoch, 'global_step': state.global_step}
                row_dict.update(kwargs['metrics'])
                csv_writer.writerow(row_dict)


def compute_belief_state_metrics(references: List[Dict[str, Dict[str, str]]],
                                 predictions: List[Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, float]]:
    metrics = {}

    # Compute domain level metrics
    ref_domains = [set(ref.keys()) for ref in references]
    pred_domains = [set(pred.keys()) for pred in predictions]

    for domain in DOMAIN_SLOTS:
        y_true_domain = [domain in ref for ref in ref_domains]
        y_pred_domain = [domain in pred for pred in pred_domains]

        precision, recall, f1, support = precision_recall_fscore_support(y_true=y_true_domain, y_pred=y_pred_domain,
                                                                         average='binary', zero_division=0)
        metrics[domain] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support,
        }

    global_y_true = []  # Accumulate all true values for global metrics
    global_y_pred = []  # Accumulate all predicted values for global metrics

    # Compute domain-slot-value level metrics
    for domain, slots in DOMAIN_SLOTS.items():
        for slot in slots:
            y_true = []
            y_pred = []
            for ref, pred in zip(references, predictions):
                ref_slot_value = ref.get(domain, {}).get(slot)
                pred_slot_value = pred.get(domain, {}).get(slot)

                if ref_slot_value is not None or pred_slot_value is not None:
                    if ref_slot_value is None:
                        # Handle case where slot is missing in reference but present in prediction
                        ref_slot_value = "missing_slot"
                    if pred_slot_value is None:
                        # Handle case where slot is missing in prediction but present in reference
                        pred_slot_value = "missing_slot"

                    y_true.append(ref_slot_value)
                    y_pred.append(pred_slot_value)

            global_y_true.extend(y_true)  # Add to global true values
            global_y_pred.extend(y_pred)  # Add to global predicted values

            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='micro',
                                                                             zero_division=0)
            metrics[f'{domain}-{slot}'] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support,
            }

    # Compute global metrics
    precision, recall, f1, support = precision_recall_fscore_support(global_y_true, global_y_pred, average='micro',
                                                                     zero_division=0)
    metrics['global'] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'support': support,
    }

    return metrics


def compute_action_metrics(references: List[List[str]], predictions: List[List[str]]) -> dict:
    # Create a MultiLabelBinarizer object
    mlb = MultiLabelBinarizer()

    # Fit and transform the MultiLabelBinarizer object with the references
    binary_references = mlb.fit_transform(references)

    # Transform the predictions with the same MultiLabelBinarizer object
    binary_predictions = mlb.transform(predictions)

    metrics = {}

    # Compute per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true=binary_references, y_pred=binary_predictions, average=None, zero_division=0)

    # Assign class metrics to action names
    for action, precision, recall, f1, support in zip(mlb.classes_, precision_per_class, recall_per_class, f1_per_class, support_per_class):
        metrics[action] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support,
        }

    # Compute global metrics with all relevant averages
    for average in ['micro', 'macro', 'weighted']:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=binary_references, y_pred=binary_predictions,
                                                                   average=average, zero_division=0)
        support = sum(support_per_class)
        metrics[f'global_{average}'] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support,
        }

    return metrics


def compute_belief_state_exact_match_ratio(references: List[Dict[str, Dict[str, str]]],
                                           predictions: List[Dict[str, Dict[str, str]]]) -> float:
    exact_match_ratio = sum(ref == pred for ref, pred in zip(references, predictions)) / len(references)
    return exact_match_ratio


def compute_actions_exact_match_ratio(references: List[List[str]], predictions: List[List[str]]) -> float:
    return sum(sorted(ref) == sorted(pred) for ref, pred in zip(references, predictions)) / len(references)



def compute_actions_metrics_classification(eval_predictions: EvalPrediction):
    logits, references = eval_predictions.predictions, eval_predictions.label_ids
    predictions = (logits >= 0).astype(np.float32)
    return {"accuracy": accuracy_score(y_true=references, y_pred=predictions),
            "recall_weighted": recall_score(y_true=references, y_pred=predictions, average='weighted'),
            "precision_weighted": precision_score(y_true=references, y_pred=predictions, average='weighted'),
            "f1_weighted": f1_score(y_true=references, y_pred=predictions, average='weighted'),
            "recall_macro": recall_score(y_true=references, y_pred=predictions, average='macro'),
            "precision_macro": precision_score(y_true=references, y_pred=predictions, average='macro'),
            "f1_macro": f1_score(y_true=references, y_pred=predictions, average='macro')
            }


def action_generation_metrics_builder(tokenizer):
    def compute_metrics(eval_predictions: EvalPrediction):
        """
        Compute metrics for action generation.
        Args:
            eval_predictions:

        Returns:

        """
        # Normally, eval_predictions.predictions are logits that need to be converted to predictions by argmax, but this
        # is already done in the preprocess_logits_for_metrics.
        predictions, label_ids = eval_predictions.predictions, eval_predictions.label_ids
        predictions[predictions == -100] = tokenizer.pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        predictions_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        predictions_list = [str_to_action_list(pred_str) for pred_str in predictions_str]
        references_list = [str_to_action_list(ref_str) for ref_str in references_str]

        metrics = compute_action_metrics(references=references_list, predictions=predictions_list)
        exact_match_ratio = compute_actions_exact_match_ratio(references=references_list,
                                                              predictions=predictions_list)

        return {
            "precision_micro": metrics['global_micro']['precision'],
            "recall_micro": metrics['global_micro']['recall'],
            "f1-score_micro": metrics['global_micro']['f1-score'],
            "precision_macro": metrics['global_macro']['precision'],
            "recall_macro": metrics['global_macro']['recall'],
            "f1-score_macro": metrics['global_macro']['f1-score'],
            "precision_weighted": metrics['global_weighted']['precision'],
            "recall_weighted": metrics['global_weighted']['recall'],
            "f1-score_weighted": metrics['global_weighted']['f1-score'],
            "exact_match_ratio": exact_match_ratio,
        }

    return compute_metrics


def belief_update_metrics_builder(tokenizer):
    def compute_metrics(eval_predictions: EvalPrediction):
        """
        Compute metrics for belief state update.
        Args:
            eval_predictions:

        Returns:

        """
        # Normally, eval_predictions.predictions are logits that need to be converted to predictions by argmax, but this
        # is already done in the preprocess_logits_for_metrics.
        predictions, label_ids = eval_predictions.predictions, eval_predictions.label_ids
        predictions[predictions == -100] = tokenizer.pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        predictions_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        predictions_dict = [str_to_belief_state(pred_str) for pred_str in predictions_str]
        references_dict = [str_to_belief_state(ref_str) for ref_str in references_str]

        metrics = compute_belief_state_metrics(references=references_dict, predictions=predictions_dict)
        exact_match_ratio = compute_belief_state_exact_match_ratio(references=references_dict,
                                                                   predictions=predictions_dict)

        return {
            "precision": metrics['global']['precision'],
            "recall": metrics['global']['recall'],
            "f1-score": metrics['global']['f1-score'],
            "exact_match_ratio": exact_match_ratio,
        }

    return compute_metrics


def preprocess_logits_for_metrics(logits: Tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor):
    """
    The Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids
