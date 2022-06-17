import argparse
import json

from seq2seq_dataset import DialogDataset, DialogDataLoader
from pathlib import Path
import torch
from torch import nn
import numpy as np
from seq2seq_dataset import PAD, UNK, SOS, EOS
from transformers import RobertaModel
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--embedding_size", default=256, type=int, help="Token embedding dimension.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--num_heads", default=6, type=int, help=".")
parser.add_argument("--num_encoder_layers", default=1, type=int, help="")
parser.add_argument("--num_decoder_layers", default=1, type=int, help="")
parser.add_argument("--dim_feedforward", default=2048, type=int, help="")
parser.add_argument("--dropout", default=0.1, type=float, help="")
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
        self.linear = nn.Linear(self.embedding_size, self.num_actions)
        self.dropout = nn.Dropout(self.dropout_rate)

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
        embedded_source = self.dropout(self.word_embedding(source_sequence)['last_hidden_state'])
        embedded_target = self.dropout(self.action_embedding(target_action))

        transformer_output = self.transformer(embedded_source, embedded_target,
                                              tgt_mask=target_mask)

        logits = self.linear(transformer_output)

        # decoder_outputs = decoder_logits.argmax(dim=2)
        # `decoder_outputs` shape: (batch_size, self.max_actions)

        return logits


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_fn: nn.CrossEntropyLoss, pad: int):
    # `decoder_logits` shape: (batch_size, predicted_seq_len, self.num_actions)
    # `target_actions` shape: (batch_size, actions_seq_len).

    batch_size = logits.shape[0]
    # Compute the bigger length from decoder outputs and targets. We use it to pad shorter sequence by zeros in order
    # to compute loss
    bigger_len = max(logits.shape[1], targets.shape[1])
    target_for_loss = torch.nn.functional.pad(targets, [0, bigger_len - targets.shape[1], 0, 0], value=pad)
    logits_for_loss = torch.nn.functional.pad(logits, [0, 0, 0, bigger_len - logits.shape[1], 0, 0], value=pad)

    loss = loss_fn(logits_for_loss.reshape(-1, logits_for_loss.shape[2]), target_for_loss.reshape(-1))

    return loss


def compute_f1(logits: torch.Tensor, targets: torch.Tensor, ids_to_action: dict, pad: int = PAD):
    # `logits` shape: (batch_size, predicted_len, num_actions)
    # `targets` shape: (batch_size, target_len).

    num_actions = len(ids_to_action)
    correct_count = {a: 0 for a in range(num_actions)}
    target_count = {a: 0 for a in range(num_actions)}
    pred_count = {a: 0 for a in range(num_actions)}

    predictions = torch.argmax(logits, dim=2)
    for a in range(num_actions):
        target_a = torch.any(torch.eq(targets, a), dim=1)
        pred_a = torch.any(torch.eq(predictions, a), dim=1)

        target_count[a] += torch.sum(target_a).detach().cpu().float().item()
        pred_count[a] += torch.sum(pred_a).detach().cpu().float().item()
        correct_count[a] += torch.sum(target_a * pred_a).detach().cpu().float().item()

    recall = {a: correct_count[a] / target_count[a] if target_count[a] > 0 else 0 for a in range(num_actions)}
    precision = {a: correct_count[a] / pred_count[a] if pred_count[a] > 0 else 0 for a in range(num_actions)}
    f1_score = {a: 2 * precision[a] * recall[a] / (precision[a] + recall[a]) if (precision[a] + recall[a]) > 0 else
                0 for a in range(num_actions)}

    average_f1 = np.mean([v for a, v in f1_score.items() if a not in [PAD, UNK, SOS, EOS]])

    return average_f1, recall, precision, f1_score


def get_pad_mask(data: torch.tensor, pad_token: int) -> torch.tensor:
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


def get_target_mask(size: int) -> torch.tensor:
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


def train(dataloader: DialogDataLoader, model: Transformer, optimizer, loss_fn: nn.CrossEntropyLoss):
    model.train()
    total_loss = 0
    with tqdm(dataloader, unit='batch') as iterations:
        for num_batch, batch in enumerate(iterations):
            source_sequence = batch['sequence'].to(device)
            target_action = batch['system_actions'].to(device)

            # Append <SOS> symbol index at the beginning of each target action sequence
            target_action = torch.cat(
                [torch.tensor([[SOS]] * target_action.size(dim=0), device=device), target_action], 1)

            # The target tensor we give as an input to the transformer must be shifted by one to the right (compared
            # to the target output tensor). In other words, the tensor we want to give the model for training must
            # have one extra element at the beginning and one less element at the end, and the tensor we compute the
            # loss function with must be shifted in the other direction.
            target_input = target_action[:, :-1]
            target_expected = target_action[:, 1:]

            # Compute target mask
            target_mask = get_target_mask(target_input.size(1)).to(device)

            # Compute padding masks
            # TODO: změnit ten pad token na nějakou proměnnou. Roberta má pad token 1.
            source_pad_mask = get_pad_mask(source_sequence, pad_token=1).to(device)
            target_pad_mask = get_pad_mask(target_input, pad_token=PAD).to(device)

            logits = model(source_sequence, target_input, target_mask, source_pad_mask, target_pad_mask)

            loss = compute_loss(logits, target_expected, loss_fn, PAD)

            # Do backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            optimizer.step()

            total_loss += loss.detach().item()

            iterations.set_postfix(loss=loss.detach().item())

    total_loss /= len(dataloader)

    return total_loss


def evaluate(dataloader: DialogDataLoader, model: Transformer, loss_fn: nn.CrossEntropyLoss):
    model.eval()
    ids_to_action = dataloader.ids_to_action
    total_loss = 0.
    total_average_f1 = 0.
    total_recall = {ids_to_action[a]: 0. for a in ids_to_action.keys()}
    total_precision = {ids_to_action[a]: 0. for a in ids_to_action.keys()}
    total_f1_score = {ids_to_action[a]: 0. for a in ids_to_action.keys()}
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
        target_mask = get_target_mask(target_input.size(1)).to(device)

        # Compute padding masks
        # TODO: změnit ten pad token na nějakou proměnnou. Roberta má pad token 1.
        source_pad_mask = get_pad_mask(source_sequence, pad_token=1).to(device)
        target_pad_mask = get_pad_mask(target_input, pad_token=PAD).to(device)

        logits = model(source_sequence, target_input, target_mask, source_pad_mask, target_pad_mask)

        loss = compute_loss(logits, target_expected, loss_fn, PAD)
        average_f1, recall, precision, f1_score = compute_f1(logits, target_expected, ids_to_action)

        total_loss += loss.detach().item()
        total_average_f1 += average_f1
        for action_idx, action in ids_to_action.items():
            total_recall[action] += recall[action_idx]
            total_precision[action] += precision[action_idx]
            total_f1_score[action] += f1_score[action_idx]

    total_loss /= len(dataloader)
    total_average_f1 /= len(dataloader)
    for action in ids_to_action.values():
        total_recall[action] /= len(dataloader)
        total_precision[action] /= len(dataloader)
        total_f1_score[action] /= len(dataloader)

    return total_loss, total_average_f1, total_recall, total_precision, total_f1_score


def fit(model: Transformer, train_dataloader: DialogDataLoader, val_dataloader: DialogDataLoader,
        optimizer: torch.optim.Optimizer, loss_fn: nn.CrossEntropyLoss, epochs: int, save_path: Path,
        max_not_improved: int = 10):
    # Used for plotting later on
    train_loss_list, val_loss_list = [], []
    history = []
    best_average_f1 = -1.0
    not_improved = 0

    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)
        train_loss = train(train_dataloader, model, optimizer, loss_fn)
        train_loss_list += [train_loss]

        val_loss, val_average_f1, val_recall, val_precision, val_f1_score = evaluate(val_dataloader, model, loss_fn)
        val_loss_list += [val_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation average f1: {val_average_f1:.4f}")
        print(f"Validation recall: {val_recall:.4f}")
        print(f"Validation precision: {val_precision:.4f}")
        print(f"Validation f1_score: {val_f1_score:.4f}")

        history.append({'epoch': epoch + 1,
                        'val_loss': val_loss,
                        'val_average_f1': val_average_f1})
        pd.DataFrame(history).to_csv(save_path / 'history.csv')

        # If the model improved then save it as the best model
        if val_average_f1 > best_average_f1:
            print(f'New model found with average f1 score {val_average_f1}, saving this model!')
            torch.save(model, save_path / 'weights.pth')
            torch.save(model.state_dict(), save_path / 'state_dict.pt')
            best_average_f1 = val_average_f1
            not_improved = 0

        else:
            not_improved += 1
            if not_improved >= max_not_improved:
                break

        print()

    return train_loss_list, val_loss_list


def main(args):
    print("Using {} device".format(device))

    save_path = Path(args.save_folder)
    if not save_path.exists():
        print(f"Save folder doesn't exist, creating it. {save_path}")
        os.makedirs(save_path)

    train_data = DialogDataset(dataset_type='train', k=10, domains=['taxi'])
    val_data = DialogDataset(dataset_type='val', k=10, domains=['taxi'])
    # val_data = train_data
    # test_data = DialogDataset(dataset_type='test', k=10, domains=['restaurant', 'hotel'])
    # train_data = DialogDataset(dataset_type='dummy', k=10)
    # val_data = DialogDataset(dataset_type='dummy', k=10)

    train_dataloader = DialogDataLoader(train_data, batch_size=args.batch_size, batch_first=True)
    action_map = train_dataloader.action_to_ids
    num_actions = train_dataloader.num_actions
    vocab_size = train_dataloader.tokenizer.vocab_size

    # TODO: save also action map into save_path

    print(f'{num_actions=}, {vocab_size=}, {len(train_data)=}')

    val_dataloader = DialogDataLoader(val_data, action_map=action_map, batch_size=args.batch_size, batch_first=True)

    if args.train_model:
        model = Transformer(embedding_size=args.embedding_size, vocabulary_size=vocab_size,
                            num_actions=num_actions,
                            num_heads=args.num_heads, num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_decoder_layers, dim_feedforward=args.dim_feedforward,
                            dropout=args.dropout).to(device)

        # Save model configuration
        (save_path / 'model_args.json').write_text(json.dumps(vars(args), indent=2))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        epochs = args.epochs

        print('TRAINING!')
        print('==============================================================================================')
        print(f'Model with configuration {vars(args)}')
        print('==============================================================================================')
        print(model)
        print('==============================================================================================')
        train_loss_list, val_loss_list = fit(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs,
                                             save_path=save_path)

        print('TRAINING DONE!\n\n')

    else:
        assert save_path.exists(), f'There is no {save_path} folder!'
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    model = torch.load(save_path / 'weights.pth', map_location=device)
    print('EVALUATING!')
    print('==============================================================================================')
    print(f'Model with configuration {vars(args)}')
    print('==============================================================================================')
    print(model)
    print('==============================================================================================')

    val_loss, val_average_f1, val_recall, val_precision, val_f1_score = evaluate(val_dataloader, model, loss_fn)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation average f1: {val_average_f1:.4f}")
    print(f"Validation recall: {val_recall:.4f}")
    print(f"Validation precision: {val_precision:.4f}")
    print(f"Validation f1_score: {val_f1_score:.4f}")
