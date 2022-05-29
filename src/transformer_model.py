import argparse

from seq2seq_dataset import DialogDataset, DialogDataLoader
from pathlib import Path
import torch
from torch import nn
import numpy as np
from seq2seq_dataset import PAD, UNK, SOS, EOS
from transformers import RobertaModel
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--token_embedding_size", default=256, type=int, help="Token embedding dimension.")
parser.add_argument("--action_embedding_size", default=128, type=int, help="Action embedding dimension.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--hidden_size", default=256, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--save_folder", default="/home/safar/HCN/models", type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
args = parser.parse_args([] if "__file__" not in globals() else None)

device = "cuda" if torch.cuda.is_available() else "cpu"


class Transformer(nn.Module):
    def __init__(self, embedding_size: int, vocabulary_size: int, num_actions: int,
                 num_heads: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float,
                 max_actions: int = 5, batch_first: bool = True, norm_first: bool = False, seq_pad_index: int = 1):
        super(Transformer, self).__init__()
        self.embedding_size = 768
        self.vocabulary_size = vocabulary_size
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.max_actions = max_actions
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.seq_pad_index = seq_pad_index

        # Embedding layer for input sequence (into Encoder) as the last layer of RoBERTa model
        self.word_embedding = RobertaModel.from_pretrained("roberta-base")

        # Positional embeddings are probably part of Roberta
        # TODO: try positional embeddings

        # Embedding for actions on the Decoder input
        self.action_embedding = nn.Embedding(self.num_actions, self.embedding_size, padding_idx=PAD)
        # TODO: maybe again positional embeddings

        self.transformer = nn.Transformer(
            d_model=self.embedding_size,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            batch_first=self.batch_first,
            # norm_first=self.norm_first
        )

        # Linear layer used after transformer which maps vector of the size `embedding_size` into
        # a vector of size `num_actions`
        self.linear = nn.Linear(self.embedding_size, num_actions)
        self.dropout = nn.Dropout(self.dropout_rate)

    def get_target_mask(self, size: int) -> torch.tensor:
        """
        Generates a square matrix (tensor) to mask targets in training which are in the future,
        tensor will be made up of zeros in the positions where the transformer can have access to the elements,
        and minus infinity where it can’t.
            # EX for size=5:
            # [[0., -inf, -inf, -inf, -inf],
            #  [0.,   0., -inf, -inf, -inf],
            #  [0.,   0.,   0., -inf, -inf],
            #  [0.,   0.,   0.,   0., -inf],
            #  [0.,   0.,   0.,   0.,   0.]]
        :param size:
        :return:
        """
        # Lower triangular matrix of boolean values. Lower triangle matrix is True.
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        # Convert zeros to -inf
        mask = mask.masked_fill(mask == 0, float('-inf'))
        # Convert ones to 0
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask

    def get_pad_mask(self, data: torch.tensor, pad_token: int) -> torch.tensor:
        """
        Creates a tensor to specify which elements are padding tokens and which are not. Binary matrix
            us used where there is a True value on the positions where the padding token is and False where it isn’t.
                If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
                [False, False, False, True, True, True]
        :param data: input data to mask
        :param pad_token: which token index is padding token
        :return:
        """

        return data == pad_token

    def forward(self, source_sequence: torch.Tensor, target_action: torch.Tensor, target_mask: torch.Tensor = None,
                source_pad_mask: torch.Tensor = None, target_pad_mask: torch.Tensor = None):
        """

        :param source_sequence: source sequence on the input of the Transformer encoder, shape: (batch_size, seq_len)
        :param target_action: actions on the input of the Decoder, shape: (batch_size, actions_seq_len).
        :param target_mask: tensor to mask targets in training which are in the future, tensor will be made up of zeros
            in the positions where the transformer can have access to the elements, and minus infinity where it can’t,
            shape: (target_len, target_len)
        :param source_pad_mask: tensor to specify which elements are padding tokens and which are not. Binary matrix
            us used where there is a True value on the positions where the padding token is and False where it isn’t.
            shape: (batch_size, sequence_len)
        :param target_pad_mask: the same as `source_pad_mask` but for target actions. Shape: (batch_size, target_len)
        :return:
        """
        batch_size = source_sequence.shape[0]
        target_action_len = target_action.shape[1]

        # Embed source sequence and target actions
        embedded_source = self.dropout(self.word_embedding(source_sequence))
        embedded_target = self.dropout(self.action_embedding(target_action))

        transformer_output = self.transformer(embedded_source, embedded_target,
                                              tgt_mask=target_mask,
                                              src_key_padding_mask=source_pad_mask,
                                              tgt_key_padding_mask=target_pad_mask)

        logits = self.linear(transformer_output)

        # decoder_outputs = decoder_logits.argmax(dim=2)
        # `decoder_outputs` shape: (batch_size, self.max_actions)

        return logits


def train(dataloader: DialogDataLoader, model: Transformer, optimizer, loss_fn):
    num_batches = len(dataloader)
    model.train()
    for num_batch, batch in enumerate(dataloader):
        source_sequence = batch['sequence'].to(device)
        target_action = batch['system_actions'].to(device)

        # Append <SOS> symbol index at the beginning of each target action sequence
        target_action = torch.cat(
            [torch.tensor([[SOS]] * target_action.size(dim=0), device=device), target_action], 1)

        # The target tensor we give as an input to the transformer must be shifted by one to the right (compared to
        # the target output tensor). In other words, the tensor we want to give the model for training must have one
        # extra element at the beginning and one less element at the end, and the tensor we compute the loss function
        # with must be shifted in the other direction.
        target_input = target_action[:, :-1]
        target_expected = target_action[:, 1:]

        # Compute target mask
        target_mask = model.get_target_mask(target_input.size(1))

        # Compute padding masks
        # TODO: změnit ten pad token na nějakou proměnnou. Roberta má pad token 1.
        source_pad_mask = model.get_pad_mask(source_sequence, pad_token=1)
        target_pad_mask = model.get_pad_mask(target_input, pad_token=PAD)

        logits = model(source_sequence, target_input, target_mask, source_pad_mask, target_pad_mask)

        loss = loss_fn(logits, target_expected)

        # Do backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num_batch % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")