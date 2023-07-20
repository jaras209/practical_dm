import argparse
import logging
from pathlib import Path

from huggingface_multiwoz_dataset import MultiWOZDatasetBeliefUpdate
from train_t5_generation import train


def main(args):
    """
    Main function to train and evaluate the model using the given arguments.

    Args:
        args: A Namespace object containing the parsed command-line arguments.
    """
    # Load the dataset
    belief_state_dataset = MultiWOZDatasetBeliefUpdate(tokenizer_name=args.tokenizer_name,
                                                       max_source_length=args.max_source_length,
                                                       max_target_length=args.max_target_length,
                                                       root_cache_path=args.data_path,
                                                       root_database_path=args.data_path,
                                                       domains=args.domains,
                                                       subset_size=args.train_subset_size)

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
    else:
        # Use the provided path for evaluation
        trained_model_path = Path(args.model_root_path) / args.model_name_or_path

    # Evaluate the model
    # logging.info("Evaluating the model...")
    # evaluate(dataset=belief_state_dataset, model_path=trained_model_path, max_target_length=args.max_target_length, )
    # logging.info("Model evaluation complete. Results saved to files.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default='google/flan-t5-base',
                        # default='flan-t5-base-finetuned-2023-06-27-10-23-50',
                        help="Name of the HuggingFace model or path from model_root_path to the pretrained model.")
    parser.add_argument("--model_root_path", type=str,
                        # default="/home/safar/HCN/models/belief_state_update",
                        default="../models/state",
                        help="Name of the folder where to save the model or where to load it from")
    parser.add_argument("--local_model", dest='local_model', action='store_true', default=False,
                        help="True indicates that we should load a locally saved model. False means that a HuggingFace "
                             "model is used for training.")
    parser.add_argument("--tokenizer_name", default='google/flan-t5-base', type=str,
                        help="Path to the pretrained Hugging face tokenizer.")
    parser.add_argument("--train_subset_size", default=0.5, type=float, help="Size of the subset of train data to use")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--max_source_length", default=260, type=int, help="Max seq length of input to model")
    parser.add_argument("--max_target_length", default=230, type=int, help="Max seq length of output to model")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--early_stopping_patience", default=15, type=int, help="Number of epochs after which the "
                                                                                "training is ended if there is no "
                                                                                "improvement on validation data")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of steps for the warmup phase. During "
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
    parser.add_argument("--metric_for_best_model", type=str, default='f1-score', help="Metric to monitor for early "
                                                                                      "stopping and saving the best "
                                                                                      "model.",
                        choices=['f1-score', 'precision', 'recall', 'exact_match_ratio'])
    parser.add_argument("--data_path",
                        # default="/home/safar/HCN/data/huggingface_data",
                        default="../data",
                        type=str,
                        help="Name of the folder where to save extracted multiwoz dataset for faster preprocessing.")
    parser.add_argument("--domains", default=[], nargs='*')
    parser.add_argument('--train', dest='train_model', action='store_true')
    parser.add_argument('--test', dest='train_model', action='store_false')
    parser.set_defaults(train_model=True)
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args)
