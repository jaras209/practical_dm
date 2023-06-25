from pathlib import Path
import csv
from typing import Tuple

import numpy as np
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, EvalPrediction
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from evaluate import load


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
        # Normally, eval_predictions.predictions are logits that need to be converted to predictions by argmax, but this
        # is already done in the preprocess_logits_for_metrics.
        predictions, references = eval_predictions.predictions, eval_predictions.label_ids
        predictions[predictions == -100] = tokenizer.pad_token_id
        return {"accuracy": accuracy_score(y_true=references, y_pred=predictions)}

    return compute_belief_metrics


def preprocess_logits_for_metrics(logits: Tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor):
    """
    The Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels
