import argparse
from dataset_torch import DialogDataset, DialogDataLoader
import torch
from torch import nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--emb_dim", default=64, type=int, help="Character embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_file", default="cz_sk_train.csv", type=str, help="Name of the train data file.")
parser.add_argument("--val_file", default="cz_sk_val.csv", type=str, help="Name of the val data file.")
parser.add_argument("--save_folder", default="save_folder", type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
args = parser.parse_args([] if "__file__" not in globals() else None)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits