import argparse
import logging
from pathlib import Path

from huggingface_multiwoz.constants import SPECIAL_TOKENS
from huggingface_multiwoz.evaluating import evaluate
from huggingface_multiwoz.huggingface_multiwoz_dataset import MultiWOZDataset
from huggingface_multiwoz.training import train


def main(args):
    """
    Main function to train and evaluate the model using the given arguments.

    Args:
        args: A Namespace object containing the parsed command-line arguments.
    """
    # Load the dataset
    multiwoz_dataset = MultiWOZDataset(tokenizer_name=args.tokenizer_name,
                                       label_column='actions',
                                       use_columns=['actions', 'utterance'],
                                       max_seq_length=args.max_seq_length,
                                       additional_special_tokens=SPECIAL_TOKENS,
                                       data_path=args.data_path,
                                       domains=args.domains)

    if args.train_model:
        # Train the model
        logging.info("Training the model...")
        trained_model_path = train(multiwoz_dataset=multiwoz_dataset,
                                   model_root_path=args.model_root_path,
                                   pretrained_model=args.pretrained_model,
                                   batch_size=args.batch_size,
                                   learning_rate=args.learning_rate,
                                   epochs=args.epochs,
                                   early_stopping_patience=args.early_stopping_patience,
                                   local_model=args.local_model)
        logging.info("Model training complete.")
    else:
        # Use the provided path for evaluation
        trained_model_path = Path(args.model_root_path) / args.pretrained_model

    # Evaluate the model
    logging.info("Evaluating the model...")
    evaluate(multiwoz_dataset=multiwoz_dataset, model_path=trained_model_path)
    logging.info("Model evaluation complete. Results saved to files.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model",
                        # default='/home/safar/HCN/models/hf_multiwoz_hotel/roberta-base/roberta-base_230315',
                        default='roberta-base',
                        # default='../models/hf_hotel_roberta-base_230316-151206',
                        type=str,
                        help="Name of the HuggingFace model or path from model_root_path to the pretrained model.")
    parser.add_argument("--model_root_path",
                        # default="/home/safar/HCN/models/hf_multiwoz_train",
                        default="../../models/",
                        type=str,
                        help="Name of the folder where to save the model or where to load it from")
    parser.add_argument("--local_model", dest='local_model', action='store_true', default=False,
                        help="True indicates that we should load a locally saved model. False means that a HuggingFace "
                             "model is used for training. ")
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str,
                        help="Path to the pretrained Hugging face tokenizer.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--max_seq_length", default=None, type=int, help="Max seq length of input to transformer")
    parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
    parser.add_argument("--early_stopping_patience", default=10, type=int, help="Number of epochs after which the "
                                                                                "training is ended if there is no "
                                                                                "improvement on validation data")

    parser.add_argument("--data_path",
                        # default="/home/safar/HCN/data/huggingface_data",
                        default="../../data/huggingface_data",
                        type=str,
                        help="Name of the folder where to save extracted multiwoz dataset for faster preprocessing.")
    parser.add_argument("--domains", default=[], nargs='*')
    parser.add_argument('--train', dest='train_model', action='store_true')
    parser.add_argument('--test', dest='train_model', action='store_false')
    parser.set_defaults(train_model=True)
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args)
