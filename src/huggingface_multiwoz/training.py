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

from metrics import MetricsCallback, compute_metrics
from huggingface_multiwoz_dataset import MultiWOZDatasetActions

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


def train(multiwoz_dataset: MultiWOZDatasetActions,
          model_root_path: str,
          pretrained_model: str,
          batch_size: int,
          learning_rate: float,
          epochs: int,
          early_stopping_patience: int,
          local_model: bool = False,
          strategy: str = 'steps',
          logging_steps: int = 100,
          save_steps: int = 100,
          warmup_steps: int = 100,
          metric_for_best_model: str = 'f1_macro') -> Path:
    """
    Train the model using the given arguments and dataset.

    Args:
        multiwoz_dataset (MultiWOZDatasetActions): The dataset object containing the dataset and tokenizer.
        model_root_path (str): The root directory for saving the model.
        pretrained_model (str): The name of the pre-trained model or HuggingFace model.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): The number of training epochs.
        early_stopping_patience (int): The number of evaluation steps without improvement to wait before stopping training.
        local_model (bool, optional): Whether the pre-trained model is a local model. Defaults to False.
        strategy (str, optional): The strategy to use for logging, evaluation, and saving model checkpoints during training. Defaults to 'steps'.
        logging_steps (int, optional): The number of training steps to perform between each logging operation. Defaults to 100.
        save_steps (int, optional): The number of training steps to perform between each checkpoint save. Defaults to 100.
        warmup_steps (int, optional): The number of warm-up steps for the learning rate scheduler. Defaults to 100.
        metric_for_best_model (str, optional): The metric name to use for saving the best model during training. Defaults to 'f1_macro'.

    Returns:
        Path: The directory where the trained model is saved.
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

    run_name = f'{Path(pretrained_model).name}-finetuned-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    output_path = Path(model_root_path) / run_name

    # Create a directory for tensorboard logs
    tensorboard_logs_dir = Path(output_path) / 'tensorboard-logs'

    if checkpoint:
        logging.info(f"Resuming training from checkpoint {checkpoint} using {pretrained_model} as base model.")
    else:
        logging.info(f"Starting training from scratch using {pretrained_model} as base model.")
    logging.info(f"Saving model to {output_path}.")

    num_of_devices = torch.cuda.device_count()
    logging.info(f"Using as the main device: {device}")
    logging.info(f"Number of devices: {num_of_devices}")

    # Create the model to train.
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                               num_labels=multiwoz_dataset.num_labels,
                                                               id2label=multiwoz_dataset.get_id2label(),
                                                               label2id=multiwoz_dataset.get_label2id(),
                                                               problem_type="multi_label_classification").to(device)

    # Calculate some useful values for later use in TrainingArguments.
    steps_factor = batch_size * num_of_devices
    logging_steps_calculated = int(logging_steps / steps_factor)
    save_steps_calculated = int(save_steps / steps_factor)
    warmup_steps_calculated = int(warmup_steps / steps_factor)
    eval_steps_calculated = int(save_steps / steps_factor)

    # Create TrainingArguments to access all the points of customization for the training.
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=str(output_path),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy=strategy,
        logging_strategy=strategy,
        save_strategy=strategy,
        eval_steps=eval_steps_calculated,
        logging_steps=logging_steps_calculated,
        save_steps=save_steps_calculated,
        logging_dir=str(tensorboard_logs_dir),
        learning_rate=learning_rate,
        weight_decay=0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        save_total_limit=1,
        warmup_steps=warmup_steps_calculated,
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
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
                                 MetricsCallback()])

    # Train our model using the trainer and save it after the training is complete.
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    logging.info("Training model finished!")

    # Evaluate the best model
    eval_results = trainer.evaluate()

    for key, value in eval_results.items():
        logging.info(f" {key}: {value}")

    return output_path
