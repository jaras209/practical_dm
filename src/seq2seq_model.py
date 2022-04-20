import argparse

from seq2seq_dataset import DialogDataset, DialogDataLoader
from pathlib import Path
import torch
from torch import nn
import numpy as np
from seq2seq_dataset import PAD, UNK, SOS, EOS
from transformers import RobertaModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--token_embedding_size", default=256, type=int, help="Token embedding dimension.")
parser.add_argument("--action_embedding_size", default=128, type=int, help="Action embedding dimension.")
parser.add_argument("--epochs", default=500, type=int, help="Number of epochs.")
parser.add_argument("--hidden_size", default=256, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--save_folder", default="/home/safar/HCN/models", type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
args = parser.parse_args([] if "__file__" not in globals() else None)

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int, embedding_size: int, num_layers: int,
                 bidirectional: bool = True, batch_first: bool = True):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.embedding_size = 768
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # Calculate number of directions
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer which is the last layer of RoBERTa model
        self.embedding = RobertaModel.from_pretrained("roberta-base")

        self.utterance_gru = nn.GRU(input_size=self.embedding_size, hidden_size=hidden_size, batch_first=batch_first,
                                    num_layers=num_layers, bidirectional=bidirectional)
        self.context_gru = nn.GRU(input_size=self.embedding_size, hidden_size=hidden_size, batch_first=batch_first,
                                  num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, utterance: torch.Tensor, utterance_mask: torch.Tensor,
                context: torch.Tensor, context_mask: torch.Tensor):
        """

        :param utterance: shape: (batch_size, utt_seq_len)
        :param utterance_mask: shape: (batch_size, utt_seq_len)
        :param context: shape: (batch_size, con_seq_len)
        :param context_mask: shape: (batch_size, con_seq_len)
        :return:
        """

        embedded_utterance = self.embedding(utterance, utterance_mask).last_hidden_state
        # `embedded_utterance` shape: (batch_size, utt_seq_len, embedding_size)

        embedded_context = self.embedding(context, context_mask).last_hidden_state
        # `embedded_context` shape: (batch_size, con_seq_len, embedding_size)

        encoder_utterance_output, encoder_utterance_hidden = self.utterance_gru(embedded_utterance)
        # `encoder_utterance_output` shape: (batch_size, utt_seq_len, self.num_directions * hidden_size)

        encoder_context_output, encoder_context_hidden = self.context_gru(embedded_context)
        # `encoder_context_output` shape: (batch_size, con_seq_len, self.num_directions * hidden_size)

        return encoder_utterance_output, encoder_context_output,


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, action_embedding_size: int, num_actions: int,
                 bidirectional: bool = True, batch_first: bool = True):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = action_embedding_size
        self.num_actions = num_actions
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # Calculate number of directions in the Encoder to determine correct size of the input hidden vectors
        self.num_directions = 2 if bidirectional else 1

        # Embedding for actions on the Decoder input
        self.action_embedding = nn.Embedding(num_actions, action_embedding_size, padding_idx=PAD)

        self.gru = nn.GRU(input_size=2 * self.num_directions * hidden_size + action_embedding_size,
                          hidden_size=self.num_directions * hidden_size, batch_first=batch_first)

        # Linear layer used after gru which maps vector of the size `hidden_size` into vector of size `num_actions`
        self.linear = nn.Linear(self.num_directions * hidden_size, num_actions)

        self.compute_attention_weights = nn.Sequential(
            nn.Linear(2 * self.num_directions * hidden_size + action_embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
            nn.Softmax(dim=1)
        )

    def compute_attention(self, embedded_action: torch.Tensor, encoder_output: torch.Tensor,
                          hidden: torch.Tensor):
        # ========== Compute attention weights ============
        #   1. First, we need to concatenate current decoder hidden state and current embedded input actions to each
        #   encoder output (hidden state). For this, we need to repeat decoder hidden and input action as
        #   many times as there are encoder outputs:
        seq_len = encoder_output.shape[1]
        broadcast_hidden = hidden.unsqueeze(dim=1).repeat(1, seq_len, 1)
        broadcast_embedded_action = embedded_action.repeat(1, seq_len, 1)
        # `broadcast_hidden` shape: (batch_size, seq_len, hidden_size)
        # `broadcast_embedded_action` shape: (batch_size, seq_len, action_embedding_size)

        #   2. Concatenate hidden state and current embedded input actions to each encoder output
        #   and compute attention weights:
        attention_input = torch.cat([encoder_output, broadcast_embedded_action, broadcast_hidden], dim=2)
        # `attention_input` shape:
        #   (batch_size, seq_len, 2 * self.num_directions * hidden_size + action_embedding_size)

        attention_weights = self.compute_attention_weights(attention_input)
        # `attention_weights` shape: (batch_size, seq_len, 1)

        # Now we want to create linear combination of hidden vectors for each batch example,
        # which is matrix of shape (seq_len, self.num_directions * hidden_size), with coefficients in
        # `attention_weights`, which is for each batch
        # example a vector of shape (seq_len, 1). This can be done by matrix multiplication using `torch.bmm`.
        # But we need to permute dimensions of `encoder_output`:
        #   (batch_size, seq_len, self.num_directions * hidden_size) ->
        #   (batch_size, self.num_directions * hidden_size, seq_len).
        # Then we can multiply it by `attention_weights` of shape: (batch_size, seq_len, 1) which
        # gives us attention of shape (batch_size, self.num_directions * hidden_size,  1).
        # Then we do permutation again to have one attention vector of given hidden_size for each batch.
        attention = torch.bmm(encoder_output.permute(0, 2, 1), attention_weights).permute(0, 2, 1)
        # `attention` shape: (batch_size, 1, self.num_directions * hidden_size)

        return attention

    def forward(self, input_action, hidden: torch.Tensor,
                encoder_utterance_output: torch.Tensor, encoder_context_output: torch.Tensor):
        """

        :param input_action: actions on the input of the Decoder, shape: (batch_size, 1), because we are sending on the
            input only one input action at the time
        :param hidden: hidden states of the Decoder, shape: (batch_size, self.num_directions * hidden_size)
        :param encoder_utterance_output: tensor of the utterance hidden states from the encoder at each timestep,
            shape: (batch_size, utt_seq_len, self.num_directions * hidden_size)
        :param encoder_context_output: tensor of the context hidden states from the encoder at each timestep,
            shape: (batch_size, con_seq_len, self.num_directions * hidden_size)
        :return:
        """

        embedded_action = self.action_embedding(input_action)
        # `embedded_action` shape: (batch_size, 1, action_embedding_size)

        utterance_attention = self.compute_attention(embedded_action=embedded_action,
                                                     encoder_output=encoder_utterance_output,
                                                     hidden=hidden)
        context_attention = self.compute_attention(embedded_action=embedded_action,
                                                   encoder_output=encoder_context_output,
                                                   hidden=hidden)
        # `utterance_attention` shape: (batch_size, 1, self.num_directions * hidden_size)
        # `context_attention` shape: (batch_size, 1, self.num_directions * hidden_size)

        # Now we can concatenate both attention vectors and current embedded action.
        # Remember that we have only one action in each batch, i.e. sequence length in decoder input is one.
        # It is because we predict one action at the time.
        decoder_input = torch.cat((utterance_attention, context_attention, embedded_action), dim=2)
        # `decoder_input` shape: (batch_size, 1, 2 * self.num_directions * hidden_size + action_embedding_size)

        # GRU expects hidden tensor to be of shape (seq_len, batch_size, self.num_directions * hidden_size).
        # Our hidden has shape (batch_size, self.num_directions * hidden_size), therefore, we have to add 0th dimension
        hidden = hidden.unsqueeze(dim=0)
        decoder_output, _ = self.gru(decoder_input, hidden)
        # `decoder_output` shape: (batch_size, 1, self.num_directions * hidden_size)

        # And pass decoder outputs through one linear layer to compute logits.
        logits = self.linear(decoder_output)
        # `logits` shape: (batch_size, 1, self.num_actions)

        # We also remove 1st dimension, which hase size one. This is needed for loss computation, and it looks better.
        logits = logits.squeeze(1)
        # `logits` shape: (batch_size, self.num_actions)

        return logits, decoder_output[:, 0, :]


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, max_actions: int = 5):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_actions = max_actions
        self.num_actions = self.decoder.num_actions

    def forward(self, utterance: torch.Tensor, utterance_mask: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor, target_action: torch.Tensor = None):
        """

        :param utterance: current user utterance of shape: (batch_size, utt_seq_len).
        :param utterance_mask:
        :param context: last k turns of user utterances and system responses separated by index of the special
            '</s>' symbol. shape: (batch_size, context_seq_len).
        :param context_mask:
        :param target_action: actions on the input of the Decoder, shape: (batch_size, actions_seq_len).
        :return:
        """
        batch_size = utterance.shape[0]
        training = True if target_action is not None else False

        # Call the encoder on the input utterance and context. The results are two tensors with hidden states at each
        # time step for both utterances and context
        encoder_utterance_output, encoder_context_output = self.encoder(utterance, utterance_mask, context, context_mask)
        # `encoder_utterance_output` shape: (batch_size, utt_seq_len, self.num_directions * hidden_size)
        # `encoder_context_output` shape: (batch_size, con_seq_len, self.num_directions * hidden_size)

        # Initialize target action to decoder's input as the vector of indices of the <SOS> symbols.
        decoder_input_action = torch.tensor([[SOS]] * batch_size, device=device)
        # `decoder_input_action` shape:(batch_size, 1)

        # Initialize decoder hidden state by the last hidden state of encoder utterance
        decoder_hidden = encoder_utterance_output[:, -1, :]
        # `decoder_hidden` shape: (batch_size, self.num_directions * hidden_size)

        # Initialize outputs from the decoder as tensor of indices of the <PAD> symbols, i.e. zeros.
        decoder_logits = torch.zeros(batch_size, self.max_actions, self.num_actions, device=device)
        # `decoder_logits` shape: (batch_size, self.max_actions, self.num_actions)

        # Initialize boolean vector which for each batch sequence shows whether the decoder output sequence finished
        finished = [False] * batch_size

        for t in range(self.max_actions):
            # Compute decoder outputs (output actions) for each batch in the current time step `t`
            logits, decoder_hidden = self.decoder(decoder_input_action, decoder_hidden,
                                                  encoder_utterance_output, encoder_context_output)
            # `logits` shape: (batch_size, self.num_actions)
            # `decoder_hidden` shape: (batch_size, self.num_directions * hidden_size)

            # Store logits form the current time step `t`
            decoder_logits[:, t, :] = logits

            # If training, then take next target action to decoder's input from the provided target tensor
            if training:
                decoder_input_action = target_action[:, t].unsqueeze(dim=1)

            # Otherwise, take next target action to decoder's input as the best action from the current predictions
            else:
                decoder_input_action = logits.argmax(dim=1).detach().unsqueeze(dim=1)

            # Set finished vector for given batch to True if the next target action is the index of <EOS> symbol.
            for i, element in enumerate(decoder_input_action.view(-1).detach().cpu().numpy()):
                if element == EOS:
                    finished[i] = True

            if all(finished):
                break

        decoder_outputs = decoder_logits.argmax(dim=2)
        # `decoder_outputs` shape: (batch_size, self.max_actions)

        return decoder_outputs, decoder_logits


def compute_loss(decoder_logits: torch.Tensor, target_actions: torch.Tensor, reduction: str = 'mean'):
    # `decoder_logits` shape: (batch_size, self.max_actions, self.num_actions)
    # `target_actions` shape: (batch_size, actions_seq_len).

    # TODO: možná nahradit loss_fn, která se posílá až z mainu
    criterion = nn.CrossEntropyLoss(reduction=reduction)

    batch_size = decoder_logits.shape[0]
    # Compute the bigger length from decoder outputs and targets. We use it to pad shorter sequence by zeros in order
    # to compute loss
    bigger_len = max(decoder_logits.shape[1], target_actions.shape[1])
    target_for_loss = torch.nn.functional.pad(target_actions, [0, bigger_len - target_actions.shape[1], 0, 0])
    logits_for_loss = torch.nn.functional.pad(decoder_logits, [0, 0, 0, bigger_len - decoder_logits.shape[1], 0, 0])

    loss = 0
    for i in range(batch_size):
        loss += criterion(logits_for_loss[i], target_for_loss[i])

    if reduction == 'mean':
        loss /= batch_size

    return loss


def train(dataloader: DialogDataLoader, model: Seq2Seq, optimizer):
    num_batches = len(dataloader)
    model.train()
    for num_batch, batch in enumerate(dataloader):
        print(num_batch)
        user_utterance = batch['user_utterance'].to(device)
        user_utterance_mask = batch['user_utterance_mask'].to(device)
        context = batch['context'].to(device)
        context_mask = batch['context_mask'].to(device)
        system_actions = batch['system_actions'].to(device)

        # Compute prediction and its loss
        outputs, logits = model(user_utterance, user_utterance_mask, context, context_mask, system_actions)
        loss = compute_loss(logits, system_actions)

        # Do backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num_batch % 100 == 0:
            loss, current = loss.item(), num_batch * len(user_utterance)
            print(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}]")


def test(dataloader: DialogDataLoader, model: Seq2Seq):
    num_batches = len(dataloader)
    num_actions = dataloader.num_actions
    model.eval()
    test_loss = 0
    correct_count = {a: 0 for a in range(num_actions)}
    target_count = {a: 0 for a in range(num_actions)}
    output_count = {a: 0 for a in range(num_actions)}
    with torch.no_grad():
        for batch in dataloader:
            user_utterance = batch['user_utterance'].to(device)
            user_utterance_mask = batch['user_utterance_mask'].to(device)
            context = batch['context'].to(device)
            context_mask = batch['context_mask'].to(device)
            system_actions = batch['system_actions'].to(device)

            # Compute prediction and its loss, we are NOT passing target system actions which means
            # that we are doing the prediction.
            outputs, logits = model(user_utterance, user_utterance_mask, context, context_mask)

            # TODO: možná dát system_actions.float()
            test_loss += compute_loss(logits, system_actions).item()

            for a in range(num_actions):
                system_action_a = (system_actions == a).any(dim=1)
                output_action_a = (outputs == a).any(dim=1)
                target_count[a] += torch.sum(system_action_a).float().item()
                output_count[a] += torch.sum(output_action_a).float().item()
                correct_count[a] += torch.sum(system_action_a * output_action_a).float().item()

            # print(dataloader.to_string(batch))
            # for o in outputs:
            #     print(dataloader.convert_ids_to_actions(o))

    recall = {a: correct_count[a] / target_count[a] if target_count[a] > 0 else 0 for a in range(num_actions)}
    precision = {a: correct_count[a] / output_count[a] if output_count[a] > 0 else 0 for a in range(num_actions)}
    f1_score = {a: 2 * precision[a] * recall[a] / (precision[a] + recall[a]) if (precision[a] + recall[a]) > 0 else
                0 for a in range(num_actions)}

    average_f1 = np.mean([v for a, v in f1_score.items() if a not in [PAD, UNK, SOS, EOS]])

    recall = {dataloader.ids_to_action[a]: v for a, v in recall.items()}
    precision = {dataloader.ids_to_action[a]: v for a, v in precision.items()}
    f1_score = {dataloader.ids_to_action[a]: v for a, v in f1_score.items()}

    test_loss /= num_batches

    print(f"\tavg loss: {test_loss:>8f} \n")
    print(f"recall=\n{recall}")
    print(f"precision=\n{precision}")
    print(f"f1_score=\n{f1_score}")

    return average_f1


def main():
    print("Using {} device".format(device))
    save_path = Path(args.save_folder)

    train_data = DialogDataset(dataset_type='train', k=10, domains=['taxi'])
    val_data = DialogDataset(dataset_type='val', k=10, domains=['taxi'])
    # test_data = DialogDataset(dataset_type='test', k=10, domains=['restaurant', 'hotel'])
    # train_data = DialogDataset(dataset_type='dummy', k=10)
    # val_data = DialogDataset(dataset_type='dummy', k=10)

    train_dataloader = DialogDataLoader(train_data, batch_size=args.batch_size, batch_first=True)
    action_map = train_dataloader.action_to_ids
    num_actions = train_dataloader.num_actions
    vocab_size = train_dataloader.tokenizer.vocab_size

    print(f'{num_actions=}, {vocab_size=}, {len(train_data)=}')

    val_dataloader = DialogDataLoader(val_data, action_map=action_map, batch_size=args.batch_size, batch_first=True)

    encoder = Encoder(vocabulary_size=vocab_size, hidden_size=args.hidden_size,
                      embedding_size=args.token_embedding_size, num_layers=1).to(device)
    decoder = Decoder(hidden_size=args.hidden_size, action_embedding_size=args.action_embedding_size,
                      num_actions=num_actions).to(device)

    model = Seq2Seq(encoder=encoder, decoder=decoder)

    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    epochs = args.epochs
    best_f1 = -100.0
    not_improved = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, optimizer)
        print("Test on train:")
        test(train_dataloader, model)
        print("Test on val:")
        average_f1 = test(val_dataloader, model)
        if average_f1 > best_f1:
            print(f'New model found, {average_f1=}, saving')
            torch.save(model, save_path / 'model' / 'weights.pth')
            best_f1 = average_f1
            not_improved = 0

        else:
            not_improved += 1
            if not_improved >= 500:
                break

    # Evaluate the best model once again
    print("---------------------------------------------")
    print("---------------------------------------------")
    print("Training finished, evaluate the best model:")
    model = torch.load(save_path / 'model' / 'weights.pth', map_location=device)
    average_f1 = test(val_dataloader, model)
    print(f'Best model {average_f1=}')

    print("Done!")
