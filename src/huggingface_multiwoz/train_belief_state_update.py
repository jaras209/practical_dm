import logging
import os
from datetime import datetime
from pathlib import Path

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch

from metrics import MetricsCallback, compute_belief_metrics
from huggingface_multiwoz_dataset import MultiWOZBeliefUpdate
from utils import highest_checkpoint

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(level=logging.INFO)


def train(dataset: MultiWOZBeliefUpdate,
          model_root_path: str,
          model_name_or_path: str,
          batch_size: int,
          learning_rate: float,
          epochs: int,
          early_stopping_patience: int,
          local_model: bool = False,
          strategy: str = 'epoch',
          logging_steps: int = 100,
          save_steps: int = 100,
          warmup_steps: int = 100,
          metric_for_best_model: str = 'f1_macro') -> Path:
    """
    Train the model using the given arguments and dataset.

    Args:
        dataset (MultiWOZDatasetActions): The dataset object containing the dataset and tokenizer.
        model_root_path (str): The root directory for saving and loading the model.
        model_name_or_path (str): The name or the path of the pre-trained model in model_root_path or HuggingFace model.
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
        model_name_or_path = Path(model_root_path) / model_name_or_path
    else:
        model_name_or_path = model_name_or_path

    # Check if we are resuming training from a checkpoint.
    checkpoint = None
    if local_model:
        checkpoints = list(model_name_or_path.glob("checkpoint-*"))
        if len(checkpoints) > 0:
            checkpoint = max(checkpoints, key=highest_checkpoint)

    run_name = f'{Path(model_name_or_path).name}-finetuned-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    output_path = Path(model_root_path) / run_name

    # Create a directory for tensorboard logs
    tensorboard_logs_dir = Path(output_path) / 'tensorboard-logs'

    if checkpoint:
        logging.info(f"Resuming training from checkpoint {checkpoint} using {model_name_or_path} as base model.")
    else:
        logging.info(f"Starting training from scratch using {model_name_or_path} as base model.")
    logging.info(f"Saving model to {output_path}.")

    num_of_devices = torch.cuda.device_count()
    logging.info(f"Using as the main device: {device}")
    logging.info(f"Number of devices: {num_of_devices}")

    # Create the model for fine-tuning
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

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
        # metric_for_best_model=metric_for_best_model,
        save_total_limit=1,
        warmup_steps=warmup_steps_calculated,
        # eval_accumulation_steps=2,
    )

    # Prepare model for training, i.e., this command does not train the model.
    model.train()

    # Create HuggingFace Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=dataset.dataset['train'],
                      eval_dataset=dataset.dataset['val'],
                      # compute_metrics=compute_belief_metrics,
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
