import argparse
from dataset_torch import DialogDataset, DialogDataLoader
import torch
from torch import nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--embedding_size", default=256, type=int, help="Character embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_size", default=512, type=int, help="RNN cell dimension.")
parser.add_argument("--pos_weight", default=10, type=int, help="Weight of positive examples in BCE loss.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--save_folder", default="save_folder", type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
args = parser.parse_args([] if "__file__" not in globals() else None)


device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int, embedding_size: int, num_actions: int,
                 bidirectional: bool = True, batch_first: bool = True):
        super(NeuralNetwork, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_actions = num_actions
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # Calculate number of directions
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer same for both utterances and context
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)

        self.utterance_gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=batch_first,
                                    bidirectional=bidirectional)
        self.context_gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=batch_first,
                                  bidirectional=bidirectional)

        self.linear = nn.Linear(2 * self.num_directions * hidden_size, num_actions)
        self.sigmoid = nn.Sigmoid()

    def forward(self, utterance: torch.Tensor, context: torch.Tensor):
        if not self.batch_first:
            utterance = utterance.transpose(0, 1)
            context = context.transpose(0, 1)

        batch_size = utterance.shape[0]

        utterance_embedding = self.embedding(utterance)
        context_embedding = self.embedding(context)

        gru_utterance_output, gru_utterance_hidden = self.utterance_gru(utterance_embedding)
        gru_utterance_output = gru_utterance_output[:, -1, :]
        gru_context_output, gru_context_hidden = self.context_gru(context_embedding)
        gru_context_output = gru_context_output[:, -1, :]
        features = torch.cat([gru_context_output, gru_utterance_output], dim=1)
        logits = self.linear(features)
        probabilities = self.sigmoid(logits)
        return logits, probabilities


def train(dataloader: DialogDataLoader, model: NeuralNetwork, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for num_batch, batch in enumerate(dataloader):
        user_utterance = batch['user_utterance'].to(device)
        context = batch['context'].to(device)
        system_actions = batch['system_actions'].to(device)

        # Compute prediction error
        logits, probabilities = model(user_utterance, context)
        loss = loss_fn(logits, system_actions.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num_batch % 100 == 0:
            loss, current = loss.item(), num_batch * len(user_utterance)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DialogDataLoader, model: NeuralNetwork, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_utterance = batch['user_utterance'].to(device)
            context = batch['context'].to(device)
            system_actions = batch['system_actions'].to(device)

            logits, probabilities = model(user_utterance, context)
            test_loss += loss_fn(logits, system_actions.float()).item()
            correct += (torch.round(probabilities) == system_actions).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    print("Using {} device".format(device))

    train_data = DialogDataset(dataset_type='train', k=10, domains=['restaurant'])
    val_data = DialogDataset(dataset_type='val', k=10, domains=['restaurant'])
    # test_data = DialogDataset(dataset_type='test', k=10, domains=['restaurant', 'hotel'])

    train_dataloader = DialogDataLoader(train_data, batch_size=args.batch_size, batch_first=True)
    action_map = train_dataloader.action_to_ids
    num_actions = train_dataloader.num_actions
    vocab_size = train_dataloader.tokenizer.vocab_size

    val_dataloader = DialogDataLoader(val_data, action_map=action_map, batch_size=args.batch_size, batch_first=True)

    model = NeuralNetwork(vocabulary_size=vocab_size, hidden_size=args.hidden_size, embedding_size=args.embedding_size,
                          num_actions=num_actions, bidirectional=True, batch_first=True).to(device)
    print(model)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight * torch.ones(num_actions))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = args.epochs
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(val_dataloader, model, loss_fn)
    print("Done!")

