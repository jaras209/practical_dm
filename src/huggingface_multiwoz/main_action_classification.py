import argparse
import logging
from pathlib import Path

from constants import SPECIAL_TOKENS
from evaluate_action_prediction import evaluate
from huggingface_multiwoz_dataset import MultiWOZDatasetActionsClassification
from train_action_prediction import train


def main(args):
    """
    Main function to train and evaluate the model using the given arguments.

    Args:
        args: A Namespace object containing the parsed command-line arguments.
    """
    # Load the dataset
    action_prediction_dataset = MultiWOZDatasetActionsClassification(tokenizer_name=args.tokenizer_name,
                                                                     max_seq_length=args.max_seq_length,
                                                                     additional_special_tokens=SPECIAL_TOKENS,
                                                                     root_cache_path=args.data_path,
                                                                     root_database_path=args.data_path,
                                                                     domains=args.domains,
                                                                     subset_size=args.train_subset_size)

    if args.train_model:
        # Train the model
        logging.info("Training the model...")
        trained_model_path = train(dataset=action_prediction_dataset,
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
    else:
        # Use the provided path for evaluation
        trained_model_path = Path(args.model_root_path) / args.model_name_or_path

    # Evaluate the model
    logging.info("Evaluating the model...")
    # evaluate(multiwoz_dataset=action_prediction_dataset, model_path=trained_model_path)
    logging.info("Model evaluation complete. Results saved to files.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        # default='roberta-base-finetuned-2023-06-08-17-45-19',
                        default='roberta-base',
                        # default='hf_multiwoz_restaurant/roberta-base-finetuned-2023-06-08-17-45-19_without_database_elements',
                        type=str,
                        help="Name of the HuggingFace model or path from model_root_path to the pretrained model.")
    parser.add_argument("--model_root_path",
                        default="/home/safarjar/HCN/models/action_classification_30",
                        # default="../../models/",
                        type=str,
                        help="Name of the folder where to save the model or where to load it from")
    parser.add_argument("--local_model", dest='local_model', action='store_true', default=False,
                        help="True indicates that we should load a locally saved model. False means that a HuggingFace "
                             "model is used for training. ")
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str,
                        help="Path to the pretrained Hugging face tokenizer.")
    parser.add_argument("--train_subset_size", default=0.5, type=float, help="Size of the subset of train data to use")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--max_seq_length", default=509, type=int, help="Max seq length of input to transformer")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
    parser.add_argument("--early_stopping_patience", default=15, type=int, help="Number of epochs after which the "
                                                                               "training is ended if there is no "
                                                                               "improvement on validation data")
    parser.add_argument("--warmup_steps", type=int, default=300, help="Number of steps for the warmup phase. During "
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
    parser.add_argument("--metric_for_best_model", type=str, default='f1_weighted', help="Metric to monitor for early "
                                                                                         "stopping and saving the best "
                                                                                         "model.",
                        choices=['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'precision_weighted',
                                 'recall_macro', 'recall_weighted'])
    parser.add_argument("--data_path",
                        default="/home/safarjar/HCN/data/huggingface_data",
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
