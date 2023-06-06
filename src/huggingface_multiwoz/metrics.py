from pathlib import Path
import csv

import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, EvalPrediction
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


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


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = (logits >= 0).astype(np.float32)
    return {"accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "recall_weighted": recall_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "precision_weighted": precision_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "f1_weighted": f1_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "recall_macro": recall_score(y_true=y_true, y_pred=y_pred, average='macro'),
            "precision_macro": precision_score(y_true=y_true, y_pred=y_pred, average='macro'),
            "f1_macro": f1_score(y_true=y_true, y_pred=y_pred, average='macro')
            }
