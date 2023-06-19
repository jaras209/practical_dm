import argparse
import logging
from pathlib import Path

from constants import SPECIAL_TOKENS
from evaluating import evaluate
from huggingface_multiwoz_dataset import MultiWOZBeliefUpdate
from train_belief_state_update import train


def main(args):
    """
    Main function to train and evaluate the model using the given arguments.

    Args:
        args: A Namespace object containing the parsed command-line arguments.
    """
    # Load the dataset
    belief_state_dataset = MultiWOZBeliefUpdate(tokenizer_name=args.tokenizer_name,
                                                max_source_length=args.max_source_length,
                                                max_target_length=args.max_target_length,
                                                root_cache_path=args.data_path,
                                                root_database_path=args.data_path,
                                                domains=args.domains)

    if args.train_model:
        # Train the model
        logging.info("Training the model...")
        trained_model_path = train(dataset=belief_state_dataset,
                                   model_root_path=args.model_root_path,
                                   model_name_or_path=args.model_name_or_path,
                                   batch_size=args.batch_size,
                                   learning_rate=args.learning_rate,
                                   epochs=args.epochs,
                                   early_stopping_patience=args.early_stopping_patience,
                                   local_model=args.local_model,
                                   strategy=args.strategy,
                                   logging_steps=args.logging_steps,
                                   save_steps=args.save_steps,
                                   warmup_steps=args.warmup_steps,
                                   metric_for_best_model=args.metric_for_best_model)
        logging.info("Model training complete.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='google/flan-t5-base',
                        help="Name of the HuggingFace model or path from model_root_path to the pretrained model.")
    parser.add_argument("--model_root_path", type=str, default="/home/safar/HCN/models/belief_state_update",
                        help="Name of the folder where to save the model or where to load it from")
    parser.add_argument("--local_model", dest='local_model', action='store_true', default=False,
                        help="True indicates that we should load a locally saved model. False means that a HuggingFace "
                             "model is used for training.")
    parser.add_argument("--tokenizer_name", default='google/flan-t5-base', type=str,
                        help="Path to the pretrained Hugging face tokenizer.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--max_source_length", default=512, type=int, help="Max seq length of input to model")
    parser.add_argument("--max_target_length", default=256, type=int, help="Max seq length of output to model")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
    parser.add_argument("--early_stopping_patience", default=10, type=int, help="Number of epochs after which the "
                                                                                "training is ended if there is no "
                                                                                "improvement on validation data")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of steps for the warmup phase. During "
                                                                        "this phase, the learning rate gradually "
                                                                        "increases to the initial learning rate.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps after which the training "
                                                                        "progress will be logged.")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Number of steps after which the current state of the model will be saved.")
    parser.add_argument("--strategy", type=str, default='epoch',
                        help="Strategy for model evaluation/logging and saving during training. "
                             "Could be either 'steps' or 'epoch'.",
                        choices=['steps', 'epoch'])
    parser.add_argument("--metric_for_best_model", type=str, default='accuracy', help="Metric to monitor for early "
                                                                                      "stopping and saving the best "
                                                                                      "model.",
                        choices=['accuracy'])
    parser.add_argument("--data_path",
                        default="/home/safar/HCN/data/huggingface_data",
                        # default="../../data/huggingface_data",
                        type=str,
                        help="Name of the folder where to save extracted multiwoz dataset for faster preprocessing.")
    parser.add_argument("--domains", default=[], nargs='*')
    parser.add_argument('--train', dest='train_model', action='store_true')
    parser.add_argument('--test', dest='train_model', action='store_false')
    parser.set_defaults(train_model=True)
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args)
