import json

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from transformers import RobertaTokenizer
import numpy as np

DOMAIN_NAMES = ['Restaurant', 'Attraction', 'Hotel', 'Taxi', 'Train', 'Bus', 'Hospital', 'Police']
PAD, UNK, SOS, EOS = 0, 1, 2, 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: `dataset_dir` must be changed to '/home/safar/HCN/data' when using AIC. Locally, use '../data'.
DATASET_DIR = '/home/safar/HCN/data'


# DATASET_DIR = '../data'


class DialogDataset(Dataset):
    """
    Class to load and store Dialog dataset.
    """

    def __init__(self, dataset_type: str, k: int, dataset_dir: str = DATASET_DIR, domains: list = None):
        """
        Load the dataset.

        :param dataset_type: specify dataset type - `train`, `val` or `test`
        :param k: truncate long contexts to `k` last utterances
        :param dataset_dir: directory of the dataset
        """

        """
        assert dataset_type in ['train', 'val', 'test'], \
            AssertionError(f'Wrong dataset_type \'{dataset_type}\'! Only acceptable dataset_type is \'train\', '
                           f'\'val\' or \'test\'.')
        """

        self.dataset_type = dataset_type
        self.k = k
        self.data = self.extract_data(dataset_dir, dataset_type, domains, strip_domain=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def extract_data(self, dataset_dir: str, dataset_type: str, domains: list = None, strip_domain: bool = False) -> \
            list[dict]:
        """
        Loads data from the specified directory.

        :param dataset_dir: dataset directory
        :param dataset_type: specify dataset type - `train`, `validation` or `test`
        :param domains:
        :param strip_domain:
        :return: list of dataset examples, where each example is a dictionary of the following structure:
            `{  'user_utterance': str,       # the string with the current user response
                'system_utterance': str,     # the string with the current system response
                'system_actions': list[str]  # list of system action names for the current turn
                'context': list[str],        # list of utterances preceding the current utterance
            }`
        """
        dataset_path = Path(dataset_dir) / dataset_type
        acts_path = Path(dataset_dir) / 'dialog_acts.json'
        # acts_path = Path(dataset_dir) / 'dummy_acts.json'
        dialogue_files = sorted(dataset_path.iterdir())

        # List to store the examples
        data = []

        with open(acts_path, 'r') as file:
            acts = json.load(file)

        for dialogue_file in dialogue_files:
            with open(dialogue_file, 'r') as file:
                dialogues = json.load(file)

            example = None
            for dialogue in dialogues:
                turns = dialogue['turns']
                dialogue_id = dialogue['dialogue_id']
                if domains is not None and not set(dialogue['services']).issubset(set(domains)):
                    continue

                context = []
                for i, turn in enumerate(turns):
                    utterance = turn['utterance']
                    turn_id = turn['turn_id']
                    speaker = turn['speaker']
                    actions = acts[dialogue_id][f'{i}']['dialog_act']

                    if speaker == 'USER':
                        example = {'user_utterance': utterance}

                    else:
                        act_name_slot_pairs = []
                        for act_name, values in actions.items():
                            if strip_domain:
                                act_name = '-'.join([x for x in act_name.split('-') if x not in DOMAIN_NAMES])
                            for slot_value_pair in values:
                                slot, _ = slot_value_pair
                                act_name_slot_pairs.append(f'{act_name}{"-" if slot != "none" else ""}'
                                                           f'{slot if slot != "none" else ""}')

                        example.update({
                            'system_utterance': utterance,
                            'system_actions': act_name_slot_pairs,
                            'context': context.copy(),
                        })
                        data.append(example)
                        context.append(example['user_utterance'])
                        context.append(example['system_utterance'])
                        if len(context) > self.k and len(context) >= 2:
                            context = context[1:]

        return data


class DialogDataLoader(DataLoader):
    """
    Class which implements custom DataLoader for Dialog dataset.
    """

    class BatchSampler(Sampler):
        """
        Class which implements custom BatchSampler for DialogDataLoader
        """

        def __init__(self, dataset, num_buckets, batch_size=64, seed=42):
            """
            Construct BatchSampler which is able to yield a batch of examples of a given batch size.
            It will always yield conversations with similar lengths (numbers of tokens) inside the same batch.
            It will not use the original data order, but will shuffle the examples randomly.

            :param dataset: input dataset
            :param num_buckets: number of buckets to use for splitting dataset examples by their length
                                (len(utterance) + len(context) is used)
            :param batch_size: batch size
            :param seed: seed for random shuffle
            """
            super().__init__(dataset)

            np.random.seed(seed)
            self.dataset = dataset
            self.batch_size = batch_size

            print(f'{self.batch_size=}')

            self.index_and_length = []
            self.max_len = -1

            for idx, example in enumerate(dataset):
                example_len = len(example['system_utterance']) + sum(len(x) for x in example['context'])
                self.index_and_length.append((idx, example_len))
                if example_len > self.max_len:
                    self.max_len = example_len

            self.bucket_boundaries = list(np.linspace(1, self.max_len, num_buckets + 1, dtype=np.int)[1:])
            self._buckets_min = np.array([0] + self.bucket_boundaries)
            self._buckets_max = np.array(self.bucket_boundaries + [self.max_len + 1])

        def __iter__(self):
            """
            Goes through tuples of (example_idx, example_len = len(utterance) + len(context)) and split them
            into buckets base of `example_len`.
            After that, shuffle buckets and from each bucket gather batches of the given `batch_size`.
            (one batch can be smaller).
            Finally, shuffle batches and yield them one by one.

            :return: indices of batch examples
            """
            buckets = dict()
            for idx, example_len in self.index_and_length:
                bucked_id = self.get_bucket_id(example_len)
                if bucked_id in buckets.keys():
                    buckets[bucked_id].append(idx)
                else:
                    buckets[bucked_id] = [idx]

            batches = []
            for bucket in buckets.values():
                np.random.shuffle(bucket)
                # batches += [bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]
                for i in range(0, len(bucket), self.batch_size):
                    batches.append(bucket[i:i + self.batch_size])

            np.random.shuffle(batches)
            for batch_idxs in batches:
                yield batch_idxs

        def __len__(self):
            return len(self.dataset)

        def get_bucket_id(self, example_len):
            buckets_mask = np.logical_and(np.less_equal(self._buckets_min, example_len),
                                          np.less(example_len, self._buckets_max))
            bucket_id = np.min(np.where(buckets_mask))
            return bucket_id

    def __init__(self, dataset: DialogDataset, action_map: dict = None, batch_size=64, num_buckets=20,
                 batch_first=True):
        """
        Create DataLoader.

        :param dataset: input dataset for which to construct DataLoader.
        :param batch_size: batch size
        :param num_buckets: number of buckets to use for splitting dataset examples by their length
            (len(utterance) + len(context) is used)
        """
        self.dataset = dataset
        self.batch_sampler = self.BatchSampler(dataset, num_buckets=num_buckets, batch_size=batch_size)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.batch_first = batch_first

        if action_map is None:
            unique_actions = sorted(list(set([action for example in dataset for action in example['system_actions']])))
            self.action_to_ids = {"<PAD>": PAD, "<UNK>": UNK, "<SOS>": SOS, "<EOS>": EOS}
            self.action_to_ids.update({v: k for k, v in enumerate(unique_actions, start=4)})
        else:
            self.action_to_ids = action_map

        self.ids_to_action = {v: k for k, v in self.action_to_ids.items()}
        self.num_actions = len(self.action_to_ids)

        super().__init__(dataset, batch_sampler=self.batch_sampler, collate_fn=self.collate_fn)

    def convert_actions_to_ids(self, actions, sort: bool = True):
        output = []
        for action in actions:
            output.append(self.action_to_ids.get(action, UNK))

        if sort:
            output.sort()

        output.append(self.action_to_ids.get('<EOS>'))
        return output

    def convert_ids_to_actions(self, actions: torch.Tensor, skip_special_actions: bool = False):
        output = []
        actions = actions.tolist()
        for action in actions:
            if action > 3:
                output.append(self.ids_to_action.get(action, '<UNK>'))

        return output

    def collate_fn(self, batch_samples: list) -> dict:
        """
        Function which takes batch examples, tokenizes them and converts them into ids.

        :param batch_samples: batch samples
        :return: dict with following structure:
            `batch = {
                  'user_utterance':             # tokenized user utterances (padded tensor of subword ids from the current dialogue turn) for all batch examples
                  'system_utterance':           # tokenized system utterances (padded tensor of subword ids from the current dialogue turn) for all batch examples
                  'system_actions':             # tokenized system actions
                  'context': list[list[int]],   # tokenized context (padded tensor of subword ids from all preceding dialogue turns, separated by the GPT-2 special `<|endoftext|>` token) for all batch examples
            }`
        """
        user_utterances = []
        user_utterances_masks = []
        system_utterances = []
        system_utterances_masks = []
        system_actions = []
        context = []
        context_masks = []
        for i, example in enumerate(batch_samples):
            # tokenizer returns dictionary:
            #   'input_ids': list of indexes of tokenized sentence (sentence is tokenized and then converted into ints)
            #   'attention_mask': list of ones with the same size as 'input_ids'
            # both are tensors with shape (1, num_of_tokens), therefore, we squeeze them to remove the first dim.
            tokenized_user_utt = self.tokenizer(example['user_utterance'], return_tensors="pt")
            tokenized_system_utt = self.tokenizer(example['system_utterance'], return_tensors="pt")
            tokenized_context = self.tokenizer('</s>'.join(example['context']) + '</s>', return_tensors="pt")

            # convert actions into ids as well and convert it into a tensor
            action = torch.tensor(self.convert_actions_to_ids(example['system_actions']), dtype=torch.int64)

            # append everything into corresponding lists, squeeze  tensors to remove first dim of size 1
            user_utterances.append(tokenized_user_utt['input_ids'].squeeze())
            user_utterances_masks.append(tokenized_user_utt['attention_mask'].squeeze())
            system_utterances.append(tokenized_system_utt['input_ids'].squeeze())
            system_utterances_masks.append(tokenized_system_utt['attention_mask'].squeeze())
            context.append(tokenized_context['input_ids'].squeeze())
            context_masks.append(tokenized_context['attention_mask'].squeeze())
            system_actions.append(action)

        # Now we have lists of individual tensors for our batch. We need to pad each tensor in a list to the
        # longest one in the list and create one tensor of shape (batch_size, size_of_the_longest_tensor_in_batch).
        # Utterances tensors need to be padded with value 1 (that is PAD value for RoBERTa).
        # Mask tensors need to be padded with value 0 (True/False)
        # Actions are padded with value 0 (our action_to_ids dictionary maps `<PAD>` action into `0`)
        user_utterances = torch.nn.utils.rnn.pad_sequence(user_utterances, batch_first=self.batch_first,
                                                          padding_value=1)
        user_utterances_masks = torch.nn.utils.rnn.pad_sequence(user_utterances_masks, batch_first=self.batch_first,
                                                                padding_value=0)
        system_utterances = torch.nn.utils.rnn.pad_sequence(system_utterances, batch_first=self.batch_first,
                                                            padding_value=1)
        system_utterances_masks = torch.nn.utils.rnn.pad_sequence(system_utterances_masks, batch_first=self.batch_first,
                                                                  padding_value=0)
        system_actions = torch.nn.utils.rnn.pad_sequence(system_actions, batch_first=self.batch_first, padding_value=0)
        context = torch.nn.utils.rnn.pad_sequence(context, batch_first=self.batch_first, padding_value=1)
        context_masks = torch.nn.utils.rnn.pad_sequence(context_masks, batch_first=self.batch_first, padding_value=1)

        # Construct the batch as a dictionary.
        batch = {
            'user_utterance': user_utterances.to(device),
            'user_utterance_mask': user_utterances_masks.to(device),
            'system_utterance': system_utterances.to(device),
            'system_utterance_mask': system_utterances_masks.to(device),
            'system_actions': system_actions.to(device),
            'context': context.to(device),
            'context_mask': context_masks.to(device)
        }
        return batch

    def to_string(self, batch, predictions=None):
        """
        Converts batch examples into strings.

        :param batch: batch
        :param predictions:
        :return: batch with strings instead of ids
        """
        output = {
            'user_utterance': [],
            'system_utterance': [],
            'system_actions': [],
            'context': [],
            'predicted_actions': []
        }
        # Go through individual batch examples and convert then back to string.
        for i in range(len(batch['context'])):
            user_utt_ = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(batch['user_utterance'][i], skip_special_tokens=True))

            system_utt_ = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(batch['system_utterance'][i], skip_special_tokens=True))

            context_ = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(batch['context'][i], skip_special_tokens=True))

            act_ = self.convert_ids_to_actions(batch['system_actions'][i], skip_special_actions=True)

            if predictions is not None:
                pred_act_ = self.convert_ids_to_actions(predictions[i], skip_special_actions=True)
                output['predicted_actions'].append(pred_act_)

            output['user_utterance'].append(user_utt_)
            output['system_utterance'].append(system_utt_)
            output['context'].append(context_)
            output['system_actions'].append(act_)

        return output
