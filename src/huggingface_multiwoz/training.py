import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Tuple, Union

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
from huggingface_multiwoz_dataset import MultiWOZDataset, _compute_metrics
from database import MultiWOZDatabase
from constants import *


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(level=logging.INFO)


def highest_checkpoint(file_path: Path) -> Tuple[int, Path]:
    """
    Extract the checkpoint number from the given file path and return a tuple with the checkpoint number and
    the file path.

    Args:
        file_path (Path): The file path to a checkpoint file.

    Returns:
        Tuple[int, Path]: A tuple containing the checkpoint number as an integer and the original file path.
    """
    # Use regex to find a sequence of digits at the end of the file path string.
    digit_sequence = re.findall("\d+$", str(file_path))

    # If a sequence of digits is found, convert it to an integer.
    # Otherwise, set the checkpoint number to -1.
    checkpoint_number = int(digit_sequence[0]) if digit_sequence else -1

    # Return a tuple containing the checkpoint number and the original file path.
    return checkpoint_number, file_path


def train(multiwoz_dataset: MultiWOZDataset,
          model_root_path: str,
          pretrained_model: str,
          batch_size: int,
          learning_rate: float,
          epochs: int,
          early_stopping_patience: int,
          local_model: bool = False) -> Path:
    """
    Train the model using the given arguments and dataset.

    Args:
        multiwoz_dataset (MultiWOZDataset): The dataset object containing the dataset and tokenizer.
        model_root_path (str): The path to the root directory of the saved models.
        pretrained_model (str): The name of the pre-trained model or HuggingFace model.
        batch_size (int): The size of the training and evaluation batch.
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): The number of training epochs.
        early_stopping_patience (int): The patience for early stopping.
        local_model (bool, optional): Whether the pre-trained model is a local model. Defaults to False.

    Returns:
        Path: The path where the trained model is saved.
    """
    if local_model:
        pretrained_model = Path(model_root_path) / pretrained_model
    else:
        pretrained_model = pretrained_model

    # Check if we are resuming training from a checkpoint.
    checkpoint = None
    if local_model:
        checkpoints = list(pretrained_model.glob("checkpoint-*"))
        if len(checkpoints) > 0:
            checkpoint = max(checkpoints, key=highest_checkpoint)

    run_name = f"{Path(pretrained_model).name}_{datetime.now().strftime('%y%m%d-%H%M%S')}"
    output_path = Path(model_root_path) / run_name

    if checkpoint:
        logging.info(f"Resuming training from checkpoint {checkpoint} using {pretrained_model} as base model.")
    else:
        logging.info(f"Starting training from scratch using {pretrained_model} as base model.")
    logging.info(f"Saving model to {output_path}.")

    # Create the model to train.
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                               num_labels=multiwoz_dataset.num_labels,
                                                               id2label=multiwoz_dataset.get_id2label(),
                                                               label2id=multiwoz_dataset.get_label2id(),
                                                               problem_type="multi_label_classification").to(device)

    # Create TrainingArguments to access all the points of customization for the training.
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=str(output_path),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        logging_dir=str(Path('logdir') / run_name),
        learning_rate=learning_rate,
        weight_decay=0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=epochs,
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
                      compute_metrics=_compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)])

    # Train our model using the trainer and save it after the training is complete.
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    logging.info("Training model finished!")

    return output_path
