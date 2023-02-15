import argparse
import os

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    pipelines
)
from transformers.pipelines.pt_utils import KeyDataset
from classification_dataset_2 import create_dataset, create_action_map, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", default='roberta-base', type=str, help="Name of the Hugging face model")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate.")
parser.add_argument("--run_name", default="new_model", type=str)
parser.add_argument("--early_stopping_patience", default=50, type=int)
parser.add_argument("--save_folder", default="/home/safar/HCN/models/new_model", type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
args = parser.parse_args([] if "__file__" not in globals() else None)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DOMAINS = ['taxi']


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = (logits >= 0).astype(np.float32)
    return {"accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "recall": recall_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "precision": precision_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "f1": f1_score(y_true=y_true, y_pred=y_pred, average='weighted')}


def main(args):
    # Load train/val datasets as Pandas DataFrames
    print("LOADING TRAIN AND VAL DATASETS...")
    train_df = load_data(dataset_type='train', k=5, domains=DOMAINS)
    val_df = load_data(dataset_type='val', k=5, domains=DOMAINS)

    action_to_ids, ids_to_action = create_action_map(train_df)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    train_dataset = create_dataset(train_df, action_to_ids, tokenizer)
    val_dataset = create_dataset(val_df, action_to_ids, tokenizer)
    print(train_dataset.features)
    print("LOADING TRAIN AND VAL DATASETS DONE!")
    print(f"{50 * '='}")
    if args.train_model:
        print(f"TRAINING A NEW MODEL USING {args.pretrained_model}...")
        training_args = TrainingArguments(
            run_name=args.run_name,
            output_dir=str(args.save_folder),
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            # logging_dir=str(logdir / run_name),
            learning_rate=args.learning_rate,
            weight_decay=0,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            num_train_epochs=args.epochs,
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            save_total_limit=5,
            warmup_steps=300,
        )

        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model,
                                                                   num_labels=len(action_to_ids),
                                                                   id2label=ids_to_action,
                                                                   label2id=action_to_ids,
                                                                   problem_type="multi_label_classification").to(device)

        model.train()
        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=val_dataset,
                          compute_metrics=compute_metrics,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])

        trainer.train()
        trainer.save_model()
        print("TRAINING A NEW MODEL DONE!")
        print(f"{50 * '='}")
    # =========================================================================================
    # Load model and create predictions
    print("EVALUATING MODEL...")
    output_path = Path(args.save_folder)
    model = AutoModelForSequenceClassification.from_pretrained(f'{args.save_folder}').to(device)
    model.eval()

    pipe = pipelines.pipeline(task='text-classification', model=model, tokenizer=tokenizer, top_k=len(action_to_ids),
                              device=0 if torch.cuda.is_available() else -1)

    predictions = pipe(KeyDataset(val_dataset, 'text'))

    predicted_labels = []
    y_pred = np.zeros((len(predictions), len(action_to_ids)), dtype=np.float)
    y_true = np.zeros((len(val_df), len(action_to_ids)), dtype=np.float)
    for i, prediction in enumerate(predictions):
        # i-th `prediction` is a model output for the i-th input dialogue.
        # It is a list with the following format:
        #   - len: num_of_actions
        #   - elements: dict (which we call `pred` later) with the following key-value pairs:
        #       - pred['label']: action name
        #       - pred['score']: probability of this action
        labels = []
        labels_ids = []
        for pred in prediction:
            score = round(pred['score'], 4)
            action = pred['label']
            if score >= 0.5:
                labels.append(action)
                labels_ids.append(action_to_ids[action])

        predicted_labels.append(labels)
        y_pred[i, labels_ids] = 1

    for i, true_actions in enumerate(val_df.system_actions.tolist()):
        labels_ids = [action_to_ids.get(action, 0) for action in true_actions]
        y_true[i, labels_ids] = 1

    val_df['predicted_actions'] = predicted_labels
    val_df['predicted_scores'] = predictions

    os.makedirs(output_path, exist_ok=True)
    val_df.to_csv(output_path / 'val_predictions.csv')
    # =========================================================================================
    # Evaluate model

    actions_recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    weighted_recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    macro_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall = {'metric': 'recall', 'macro': round(macro_recall, 4), 'weighted': round(weighted_recall, 4)} | \
             {action: round(actions_recall[i], 4) for i, action in ids_to_action.items()}

    actions_precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    weighted_precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    macro_precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision = {'metric': 'precision', 'macro': round(macro_precision, 4), 'weighted': round(weighted_precision, 4)} | \
                {action: round(actions_precision[i], 4) for i, action in ids_to_action.items()}

    actions_f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    weighted_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1 = {'metric': 'f1', 'macro': round(macro_f1, 4), 'weighted': round(weighted_f1, 4)} | \
         {action: round(actions_f1[i], 4) for i, action in ids_to_action.items()}

    accuracy = {'metric': 'accuracy', 'macro': round(accuracy_score(y_true=y_true, y_pred=y_pred), 4)}

    metrics_df = pd.DataFrame([accuracy, recall, precision, f1])
    print(metrics_df)
    metrics_df.to_csv(output_path / 'metrics.csv')

    print("EVALUATING MODEL DONE")
