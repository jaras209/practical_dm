import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Tuple, Union

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AddedToken,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    pipelines
)
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from transformers.pipelines.pt_utils import KeyDataset

from huggingface_multiwoz_dataset import MultiWOZDataset, compute_metrics
from database import MultiWOZDatabase
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model",
                    # default='/home/safar/HCN/models/hf_multiwoz_hotel/roberta-base/roberta-base_230315-170036_230316-124221',
                    default='roberta-base',
                    # default='../models/hf_hotel_roberta-base_230316-151206',
                    type=str,
                    help="Name of the HuggingFace model or path from model_root_path to the pretrained model.")
parser.add_argument("--tokenizer_name", default='roberta-base', type=str,
                    help="Path to the pretrained Hugging face tokenizer.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--max_seq_length", default=None, type=int, help="Max seq length of input to transformer")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--early_stopping_patience", default=10, type=int, help="Number of epochs after which the "
                                                                            "training is ended if there is no "
                                                                            "improvement on validation data")
parser.add_argument("--model_root_path",
                    # default="/home/safar/HCN/models/hf_multiwoz_train",
                    default="../../models/",
                    type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument("--data_path",
                    # default="/home/safar/HCN/data/huggingface_data",
                    default="../../data/huggingface_data",
                    type=str,
                    help="Name of the folder where to save extracted multiwoz dataset for faster preprocessing.")
parser.add_argument("--domains", default=['train'], nargs='*')
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def highest_checkpoint(f):
    s = re.findall("\d+$", str(f))
    return int(s[0]) if s else -1, f


def train(args, multiwoz_dataset: MultiWOZDataset) -> Path:
    print("Training model...")
    print(f"Using device: {device}")
    print(f"Using tokenizer: {args.tokenizer_name}")
    print(f"Using pretrained model: {args.pretrained_model}")
    print(f"Using batch size: {args.batch_size}")
    print(f"Using max sequence length: {args.max_seq_length}")
    print(f"Using epochs: {args.epochs}")
    pretrained_model_path = Path(args.model_root_path) / args.pretrained_model

    # Check if we are resuming training from a checkpoint.
    # If so, load the checkpoint and continue training.
    checkpoint = False
    checkpoints = list(pretrained_model_path.glob("checkpoint-*"))
    if len(checkpoints) > 0:
        checkpoint = max(checkpoints, key=highest_checkpoint)
        run_name = f"{pretrained_model_path.parent.name}_{datetime.now().strftime('%y%m%d-%H%M%S')}"
        output_path = pretrained_model_path.parent / run_name
        pretrained_model = pretrained_model_path
        print(f"Resuming training from checkpoint {checkpoint} using {args.pretrained_model} as base model.")
    else:
        run_name = f"{pretrained_model_path.name}_{datetime.now().strftime('%y%m%d-%H%M%S')}"
        output_path = pretrained_model_path / run_name
        pretrained_model = args.pretrained_model
        print(f"Starting training from scratch using {args.pretrained_model} as base model.")
    print(f"Saving model to {output_path}.")

    # Create the model to train.
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                               num_labels=multiwoz_dataset.num_labels,
                                                               id2label=multiwoz_dataset.id2label,
                                                               label2id=multiwoz_dataset.label2id,
                                                               problem_type="multi_label_classification").to(device)

    # Create TrainingArguments to access all the points of customization for the training.
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        logging_dir=str(Path('logdir') / run_name),
        learning_rate=args.learning_rate,
        weight_decay=0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=args.epochs,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1,
        warmup_steps=300,
    )
    # Resize the model embedding matrix to match the tokenizer, which is extended by BELIEF, CONTEXT, USER tokens.
    model.resize_token_embeddings(len(multiwoz_dataset.tokenizer))

    # Prepare model for training, i.e. this command does not train the model.
    model.train()

    # Create HuggingFace Trainer, which provides an API for feature-complete training in PyTorch for most
    # standard use cases. The API supports distributed training on multiple GPUs/TPUs, mixed precision through
    # NVIDIA Apex and Native AMP for PyTorch. The Trainer contains the basic training loop which supports the
    # above features.
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=multiwoz_dataset.dataset['train'],
                      eval_dataset=multiwoz_dataset.dataset['val'],
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])

    # Train our model using the trainer and save it after the training is complete.
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    print("Training model finished!")
    print(f"{50 * '='}")

    return output_path


def evaluate(args, multiwoz_dataset: MultiWOZDataset, model_path: Path, top_k: int = 5, only_dataset: str = None):
    """
    Evaluate the model on the datasets in multiwoz_dataset.
    :param args:
    :param multiwoz_dataset:
    :param model_path:
    :param top_k:
    :param only_dataset:
    :return:
    """
    if only_dataset is not None and only_dataset not in multiwoz_dataset.dataset:
        raise ValueError(f"Dataset {only_dataset} not found in multiwoz_dataset.")

    print(f"Evaluating model {model_path}...")

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

        print(f"Evaluating model on {dataset_name}...")
        output_df = dataset_data.to_pandas()
        eval_start_time = time.time()

        # Compute predictions using the model pipeline.
        predictions = classifier_pipeline(KeyDataset(dataset_data, 'text'))

        # Compute the time it took to compute the predictions.
        eval_time = time.time() - eval_start_time
        print(f"Predictions computed in {eval_time:.2f} seconds.")

        # Compare the predicted labels with the true labels.
        predicted_labels = []
        y_pred = np.zeros((len(predictions), len(multiwoz_dataset.label2id)), dtype='bool')
        y_true = np.array(dataset_data['label'])
        comparing_start_time = time.time()
        print(f"Comparing predictions with true labels...")
        for i, prediction in tqdm(enumerate(predictions), total=len(predictions)):
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
                    labels_ids.append(multiwoz_dataset.label2id[action])

            predicted_labels.append(labels)
            y_pred[i, labels_ids] = 1

        comparing_time = time.time() - comparing_start_time
        print(f"Comparing predictions with true labels took {comparing_time:.2f} seconds.")

        # Save predictions to file.
        y_pred = y_pred.astype('float')
        output_df['predicted'] = predicted_labels
        output_df['scores'] = predictions
        output_df = output_df[['text', 'actions', 'predicted', 'system_utterance', 'scores']]
        model_path.mkdir(exist_ok=True)
        output_df.to_csv(model_path / f'{dataset_name}_predictions.csv')
        print(f"Predictions saved to {model_path / f'{dataset_name}_predictions.csv'}.")

        # Compute metrics
        print(f"Computing metrics for {dataset_name}...")
        metrics_start_time = time.time()
        actions_recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)
        weighted_recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
        macro_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
        recall = {'metric': 'recall', 'macro': round(macro_recall, 4), 'weighted': round(weighted_recall, 4)} | \
                 {action: round(actions_recall[i], 4) for i, action in multiwoz_dataset.id2label.items()}

        actions_precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
        weighted_precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
        macro_precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
        precision = {'metric': 'precision', 'macro': round(macro_precision, 4),
                     'weighted': round(weighted_precision, 4)} | \
                    {action: round(actions_precision[i], 4) for i, action in multiwoz_dataset.id2label.items()}

        actions_f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
        weighted_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        f1 = {'metric': 'f1', 'macro': round(macro_f1, 4), 'weighted': round(weighted_f1, 4)} | \
             {action: round(actions_f1[i], 4) for i, action in multiwoz_dataset.id2label.items()}

        accuracy = {'metric': 'accuracy', 'macro': round(accuracy_score(y_true=y_true, y_pred=y_pred), 4)}

        metrics_time = time.time() - metrics_start_time
        print(f"Computing metrics for {dataset_name} took {metrics_time:.2f} seconds.")

        # Save metrics to file.
        metrics_df = pd.DataFrame([accuracy, recall, precision, f1])
        print(metrics_df)
        metrics_df.to_csv(model_path / f'{dataset_name}_metrics.csv')
        print(f"Metrics saved to {model_path / f'{dataset_name}_metrics.csv'}.")
        print(f"Evaluating model on {dataset_name} done in {time.time() - eval_start_time}.\n")
        print("=======================================================")

    print(f"Evaluating model {model_path} on all datasets done in {time.time() - start_time}.")


def main(args):
    print(f"{50 * '='}")
    print(f"TRAINING MODEL ON {args.domains} DOMAINS")
    print(f"{50 * '='}")
    print(f"Arguments: {args}")

    multiwoz_dataset = MultiWOZDataset(tokenizer_name=args.tokenizer_name,
                                       label_column='actions',
                                       use_columns=['actions', 'utterance'],
                                       max_seq_length=args.max_seq_length,
                                       additional_special_tokens=SPECIAL_TOKENS,
                                       data_path=args.data_path,
                                       domains=args.domains)

    if args.train_model:
        # Train model.
        model_path = train(args, multiwoz_dataset=multiwoz_dataset)

    else:
        model_path = Path(args.model_root_path) / args.pretrained_model

    # Evaluate model.
    evaluate(args, multiwoz_dataset=multiwoz_dataset, model_path=model_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
