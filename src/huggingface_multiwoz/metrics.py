from pathlib import Path
import csv
from typing import Tuple

import numpy as np
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, EvalPrediction
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import evaluate

from huggingface_multiwoz_dataset import str_to_belief_state


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


def compute_actions_metrics(eval_predictions: EvalPrediction):
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


def belief_compute_metrics_builder(tokenizer):
    def compute_belief_metrics(eval_predictions: EvalPrediction):
        """
        Compute metrics for belief state update.
        Args:
            eval_predictions:

        Returns:

        """
        rouge = evaluate.load('rouge')
        # Normally, eval_predictions.predictions are logits that need to be converted to predictions by argmax, but this
        # is already done in the preprocess_logits_for_metrics.
        predictions, references = eval_predictions.predictions, eval_predictions.label_ids
        predictions[predictions == -100] = tokenizer.pad_token_id
        references[references == -100] = tokenizer.pad_token_id

        predictions_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references_str = tokenizer.batch_decode(references, skip_special_tokens=True)

        # Accuracy
        num_correct_sequences = sum(np.array_equal(p, r) for p, r in zip(predictions, references))
        total_sequences = len(predictions)
        accuracy = num_correct_sequences / total_sequences

        # Rouge
        rouge_output = rouge.compute(predictions=predictions_str, references=references_str,
                                     rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
                                     )
        # Precision, recall, f1
        predictions_labels = []
        references_labels = []
        for pred_str, ref_str in zip(predictions_str, references_str):
            # Convert the string representations back into dictionary format
            pred_dict = str_to_belief_state(pred_str)
            ref_dict = str_to_belief_state(ref_str)

            # Convert the dictionaries into lists of domain-slot-value triples
            pred_triples = [(domain, slot, value) for domain, slots in pred_dict.items() for slot, value in
                            slots.items() if value != "None"]
            ref_triples = [(domain, slot, value) for domain, slots in ref_dict.items() for slot, value in
                           slots.items() if value != "None"]

            # Convert the triples into binary labels (1 for each triple that exists in the reference, 0 otherwise)
            all_triples = list(set(pred_triples + ref_triples))
            pred_triples = set(pred_triples)
            ref_triples = set(ref_triples)
            predictions_labels.extend([int(triple in pred_triples) for triple in all_triples])
            references_labels.extend([int(triple in ref_triples) for triple in all_triples])

        return {"accuracy": accuracy,
                "precision": precision_score(y_true=references_labels, y_pred=predictions_labels, zero_division=0),
                "recall": recall_score(y_true=references_labels, y_pred=predictions_labels, zero_division=0),
                "f1_score": f1_score(y_true=references_labels, y_pred=predictions_labels, zero_division=0),
                "rouge1": rouge_output["rouge1"],
                "rouge2": rouge_output["rouge2"],
                "rougeL": rouge_output["rougeL"],
                "rougeLsum": rouge_output["rougeLsum"]}

    return compute_belief_metrics


def preprocess_logits_for_metrics(logits: Tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor):
    """
    The Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids
