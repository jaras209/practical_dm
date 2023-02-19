import json
import pickle
import argparse
import os
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import torch.utils.data as torchdata
import copy
from typing import Optional, List, Any, Tuple
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    pipelines
)

DOMAIN_NAMES = {'hotel', 'restaurant', 'police', 'bus', 'train', 'attraction', 'hospital', 'taxi'}
DOMAIN_SLOTS = dict(bus={'bus-departure', 'bus-destination', 'bus-leaveat', 'bus-day'},
                    hotel={'hotel-pricerange', 'hotel-bookstay', 'hotel-bookday', 'hotel-area', 'hotel-stars',
                           'hotel-bookpeople', 'hotel-parking', 'hotel-type', 'hotel-name', 'hotel-internet'},
                    restaurant={'restaurant-bookpeople', 'restaurant-booktime', 'restaurant-pricerange',
                                'restaurant-name', 'restaurant-area', 'restaurant-bookday', 'restaurant-food',
                                },
                    police={'something_police'},
                    train={'train-departure', 'train-bookpeople', 'train-destination', 'train-leaveat',
                           'train-day', 'train-arriveby'},
                    attraction={'attraction-type', 'attraction-area', 'attraction-name'},
                    hospital={'hospital-department'},
                    taxi={'taxi-departure', 'taxi-destination', 'taxi-arriveby', 'taxi-leaveat'})

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", default='roberta-base', type=str,
                    help="Path to the pretrained Hugging face model and its tokenizer.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--max_seq_length", default=512, type=int, help="Max seq length of input to transformer")
parser.add_argument("--epochs", default=400, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--early_stopping_patience", default=20, type=int, help="Number of epochs after which the "
                                                                            "training is ended if there is no "
                                                                            "improvement on validation data")
parser.add_argument("--save_folder", default="models", type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
args = parser.parse_args([] if "__file__" not in globals() else None)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def save_data(data, f_name):
    assert not os.path.exists(f"{f_name}.json"), f"{f_name}.json already exists."
    with open(f"{f_name}.json", 'wb+') as f:
        pickle.dump(data, f)

    pd.DataFrame(data).to_csv(f"{f_name}.csv")


def parse_dialogue_into_examples(dialogue, context_len: int = None, strip_domain: bool = False) -> list[dict]:
    """
    Parses a dialogue into a list of examples.
    Each example is a dictionary of the following structure:
    {
        'context': list[str],  # list of utterances preceding the current utterance
        'utterance': str,  # the string with the current user response
        'delex_utterance': str,  # the string with the current user response which is delexicalized, i.e. slot
                                values are
                                # replaced by corresponding slot names in the text.
        'belief_state': dict[str, dict[str, str]],  # belief state dictionary, for each domain a separate belief state dictionary,
                                                    # choose a single slot value if more than one option is available
        'database_results': dict[str, int] # dictionary containing the number of matching results per domain
    }
    The context can be truncated to k last utterances.


    Existing services:
        {'hotel', 'restaurant', 'police', 'bus', 'train', 'attraction', 'hospital', 'taxi'}
    Existing intents:
        {'find_bus', 'find_train', 'find_restaurant', 'find_attraction', 'book_hotel', 'find_taxi',
        'find_police', 'book_train', 'find_hotel', 'find_hospital', 'book_restaurant'}
    Existing slots_values_names:
        {'bus-departure', 'hotel-pricerange', 'train-departure', 'hotel-bookstay', 'hotel-bookday',
        'restaurant-bookpeople', 'restaurant-booktime', 'restaurant-pricerange', 'attraction-type',
        'restaurant-name', 'bus-destination', 'train-bookpeople', 'hotel-area', 'taxi-departure',
        'taxi-destination', 'attraction-area', 'attraction-name', 'restaurant-area', 'taxi-arriveby',
        'hotel-stars', 'restaurant-bookday', 'taxi-leaveat', 'hotel-bookpeople', 'restaurant-food',
        'train-destination', 'hospital-department', 'hotel-parking', 'hotel-type', 'train-leaveat',
        'bus-leaveat', 'train-day', 'hotel-name', 'hotel-internet', 'train-arriveby', 'bus-day'}
    """

    examples = []
    turns = dialogue['turns']

    example = dict()
    belief_state = {domain: {slot: 'None' for slot in DOMAIN_SLOTS[domain]} for domain in DOMAIN_NAMES}
    for turn_id, _ in enumerate(turns['turn_id']):
        speaker = turns['speaker'][turn_id]
        utterance = turns['utterance'][turn_id]

        # USER
        if speaker == 0:
            frame = turns['frames'][turn_id]
            domains = frame['service']
            states = frame['state']

            for domain, state in zip(domains, states):
                slots = state['slots_values']['slots_values_name']
                values = state['slots_values']['slots_values_list']

                slot_value_pairs = {slot: value[0] for slot, value in zip(slots, values)}
                belief_state[domain].update(slot_value_pairs)

            example = {
                'utterance': utterance,
                'belief_state': copy.deepcopy(belief_state),
            }

        # SYSTEM
        else:
            dialogue_acts = turns['dialogue_acts'][turn_id]
            act_type_slot_name_pairs = []
            act_types = dialogue_acts['dialog_act']['act_type']
            act_slots = dialogue_acts['dialog_act']['act_slots']
            for act_type, act_slot in zip(act_types, act_slots):
                if strip_domain:
                    act_type = '-'.join([x for x in act_type.split('-') if x not in DOMAIN_NAMES])

                slot_names = act_slot['slot_name']
                slot_values = act_slot['slot_value']

                for slot_name in slot_names:
                    act_type_slot_name_pairs.append(f'{act_type}{"-" if slot_name != "none" else ""}'
                                                    f'{slot_name if slot_name != "none" else ""}')

            context = turns['utterance'][:turn_id - 1]
            if context_len is not None and len(context) > context_len:
                context = context[-context_len:]
            example.update({
                'context': context,
                'actions': list(set(act_type_slot_name_pairs)),
                'system_utterance': utterance,
            })
            examples.append(example)

    return examples


def load_multiwoz_dataset(split: str, context_len=None, cache_dir="../huggingface_dataset") -> pd.DataFrame:
    """
    Load the MultiWoz dataset using huggingface.datasets.
    Able to shorten the context length by setting context_len.
    """

    cache_dir = Path(cache_dir)
    print(Path(cache_dir).absolute())

    # Create cache dir if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    f_name = os.path.join(cache_dir, f"{split}_preprocessed_data")

    # If the dataset has already been preprocessed, load it from the cache
    if os.path.isfile(f"{f_name}.json"):
        data = pickle.load(open(f"{f_name}.json", 'rb'))
        print(f"Loaded {len(data)} examples from cached file.")

    else:
        multi_woz_dataset = datasets.load_dataset(path='multi_woz_v22', split=split, ignore_verifications=True,
                                                  streaming=True)
        data = []
        for idx, dialogue in enumerate(multi_woz_dataset):
            if idx % 500 == 0:
                print(f"Processing dialogue {idx + 1}")

            data.extend(parse_dialogue_into_examples(dialogue, context_len=context_len))

        save_data(data, f_name)

    return pd.DataFrame(data)


def convert_actions_to_ids(actions: list[str], dictionary: dict) -> list[int]:
    output = [dictionary.get(a, 0) for a in actions]
    return output


class DataModule:
    def __init__(self,
                 model_name_or_path: str,
                 label_column: str,
                 use_columns: List[str],
                 context_len: int = None,
                 max_seq_length: int = 512,
                 val_size: float = 0.3
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.label_column = label_column
        self.use_columns = use_columns
        self.max_seq_length = max_seq_length
        self.context_len = context_len
        self.val_size = val_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        train_df = load_multiwoz_dataset('train', context_len=self.context_len)
        val_df = load_multiwoz_dataset('validation', context_len=self.context_len)
        test_df = load_multiwoz_dataset('test', context_len=self.context_len)

        # Gather unique labels which are used in 'label' <-> 'integers' map
        self.unique_actions = sorted(list(set([action for example in train_df['actions'] for action in example])))
        self.unique_actions.append('<UNK_ACT>')
        self.num_labels = len(self.unique_actions)

        # Create HuggingFace datasets
        train_dataset = self.create_huggingface_dataset(train_df)
        val_dataset = self.create_huggingface_dataset(val_df)
        test_dataset = self.create_huggingface_dataset(test_df)

        # The 'label' <-> 'integers' map is saved into 'label2id' and 'id2label' dictionaries and saved as a part
        # of the model in model config file.
        self.label2id = {l: train_dataset.features['label'].str2int(l) for l in self.unique_actions}
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Create datasets dictionary
        self.dataset = datasets.DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'val': val_dataset}
        )

    def create_huggingface_dataset(self, df: pd.DataFrame) -> datasets.Dataset:
        """
        Creates HuggingFace dataset from pandas DataFrame
        :param df: input DataFrame
        :return: output HuggingFace dataset
        """
        # Create new 'label' column as the copy of 'self.label_column'. This new 'label' column
        # will later be mapped into indexes of labels.
        df['labels'] = df[self.label_column]

        # Create HuggingFace dataset from Dataset
        dataset = datasets.Dataset.from_pandas(df)

        # Casts 'label' column into 'ClassLabel' type which also maps labels into integers.
        labels_type = datasets.Sequence(datasets.ClassLabel(names=self.unique_actions))
        dataset = dataset.cast_column('labels', labels_type)

        # Map dataset using the 'tokenize_function'
        dataset = dataset.map(self.tokenize_function, batched=True, batch_size=256)

        return dataset

    def tokenize_function(self, example_batch):
        """
        This function prepares each batch for input into the transformer by tokenizing the text, mapping the
        tokenized text into numbers, and adding new arguments with the necessary tensors for input into the model.
        :param example_batch: batch
        :return: augmented batch with added features
        """
        #TODO: přidat ke vstupnimu textu také string reprezentaci belief_state.
        # Přidat nějaký token, který bude reprezentovat začátek belief_state a začátek utterancí.
        # Ty se musí přídat také do tokenizeru
        turns = (context + [utterance] for context, utterance in
                 zip(example_batch['context'], example_batch['utterance']))
        texts = list(map(lambda x: self.tokenizer.sep_token.join(x), turns))
        tokenized = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_seq_length)
        return tokenized

    def create_labels(self, example_batch):
        actions = np.zeros((len(example_batch['actions']), self.num_labels), dtype=np.float)
        for i, action_list in enumerate(example_batch['actions']):
            action_ids = convert_actions_to_ids(action_list, action_to_ids)
            actions[i, action_ids] = 1

        examples['label'] = actions


print(f"CREATING DATASET...")
"""
print(f"\t LOADING TRAIN DATA FROM \'{args.train_data}\'...\n"
      f"\t LOADING VAL DATA FROM \'{args.val_data}\'...\n"
      f"\t LOADING TEST DATA FROM \'{args.test_data}\'...\n"
      f"\t LOADING TOKENIZER FROM \'{args.pretrained_model}\'...\n")
"""
dm = DataModule(model_name_or_path=args.pretrained_model,
                label_column='actions',
                use_columns=['actions', 'utterance'],
                max_seq_length=args.max_seq_length)
