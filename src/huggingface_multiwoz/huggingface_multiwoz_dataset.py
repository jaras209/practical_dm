import pickle
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import copy
from typing import Optional, List, Any, Tuple, Union
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from database import MultiWOZDatabase
from transformers import AutoTokenizer
from constants import *


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = (logits >= 0).astype(np.float32)
    return {"accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "recall": recall_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "precision": precision_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "f1": f1_score(y_true=y_true, y_pred=y_pred, average='weighted')}


def save_data(data, file_path: Path):
    assert not file_path.exists(), f"{file_path} already exists."
    with open(f"{file_path}.json", 'wb+') as f:
        pickle.dump(data, f)

    pd.DataFrame(data).to_csv(f"{file_path}.csv")


def parse_dialogue_into_examples(dialogue, dialogue_domain: str, database: MultiWOZDatabase, context_len: int = None,
                                 strip_domain: bool = False) -> list[dict]:
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
            # Create example instance with user utterance and old belief state, i.e. belief state, which is in the
            # input to our model. It represents the belief state of our system after the previous turn and will be on
            # the output in the previous turn.
            example = {
                'utterance': utterance,
                'old_belief_state': copy.deepcopy(belief_state),
                'domain': dialogue_domain
            }
            frame = turns['frames'][turn_id]
            domains = frame['service']
            states = frame['state']

            # Update the belief state with this new user utterance
            for domain, state in zip(domains, states):
                slots = state['slots_values']['slots_values_name']
                values = state['slots_values']['slots_values_list']

                slot_value_pairs = {slot: value[0] for slot, value in zip(slots, values)}
                belief_state[domain].update(slot_value_pairs)

            # Create state update, which is a dictionary with the same structure as belief state but with only those
            # keys and values that changed from the old belief state to the new one
            state_update = dict()
            old_belief_state = example['old_belief_state']
            for domain, state in belief_state.items():
                state_update[domain] = {slot: value for slot, value in state.items() if
                                        slot not in old_belief_state.get(domain, {}) or
                                        old_belief_state[domain][slot] != value}

            # Get database results sizes for each domain
            database_results = {domain: len(database.query(domain, domain_state))
                                for domain, domain_state in belief_state.items()}

            # From the USER we use:
            #   - 'utterance': what the user said in the current turn
            #   - 'belief_state': the belief state of the user side of the conversation
            example.update({'new_belief_state': copy.deepcopy(belief_state),
                            'state_update': copy.deepcopy(state_update),
                            'database_results': copy.deepcopy(database_results)})

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
                if context_len > 0:
                    context = context[-context_len:]

                else:
                    context = []
            # From the SYSTEM we use:
            #   - 'context': the last `context_len` turns of the dialogue ending with the last system utterance.
            #                context together with user utterance and their belief state create input to the model
            #   - 'actions': the goal actions the model should predict from the input. It represents the SYSTEM's
            #                decision of what to do next
            #   - 'system_utterance': the SYSTEM's response based on the 'actions'. This is not used in our model in
            #                         any way, but it's a good idea to store it as well for manual control.
            example.update({
                'context': context,
                'actions': list(set(act_type_slot_name_pairs)),
                'system_utterance': utterance,
            })
            examples.append(example)

    return examples


def load_multiwoz_dataset(split: str, domains: List[str] = None, context_len: int = None,
                          only_single_domain: bool = False,
                          data_path: Union[Path, str] = "/home/safar/HCN/data/huggingface_data") -> pd.DataFrame:
    """
    Load the MultiWoz dataset using huggingface.datasets.
    Able to shorten the context length by setting context_len.
    """
    if domains is None:
        # If no domains are specified, we use all of them
        domains = list(DOMAIN_NAMES)

    # Sort domains to always get the same order of in the paths and file names below
    domains.sort()

    # Create cache directory path and database directory path from data path
    cache_path = Path(data_path) / "cached_datasets"
    database_path = Path(data_path) / "databases"

    cache_path = cache_path / ('-'.join(domains) + f"_only-single-domain_{only_single_domain}")
    print(f"Cache dir = {Path(cache_path).absolute()}")

    # Create cache directory, if it doesn't exist
    cache_path.mkdir(exist_ok=True)

    # Create file path for current split (train/test/val)
    file_path = cache_path / f"{split}_preprocessed_data_{'-'.join(domains)}"

    # Turn list of domains into a set to be used in subset queries in filtering dialogues containing only these domains
    domains = set(domains)

    # If the dataset has already been preprocessed, load it from the cache
    if file_path.is_file():
        data = pickle.load(open(f"{file_path}.json", 'rb'))
        print(f"Loaded {len(data)} examples from cached file.")

    # Else, load MultiWoz 2.2 dataset from HuggingFace, create data and save them into a cache directory
    else:
        # Load MultiWoz dataset from HuggingFace
        multi_woz_dataset = datasets.load_dataset(path='multi_woz_v22', split=split, ignore_verifications=True,
                                                  streaming=True)

        # Load MultiWoz Database, which is locally saved at database_path
        database = MultiWOZDatabase(database_path)

        data = []
        # Iterate through dialogues in the dataset, preprocessing each dialogue
        for dialogue in multi_woz_dataset:
            if only_single_domain and len(dialogue['services']) != 1:
                continue

            # Use only those dialogues containing allowed domains and parse them into examples
            if not set(dialogue['services']).issubset(domains):
                continue

            if len(dialogue['services']) > 0:
                dialogue_domain = dialogue['services'][0]
            else:
                dialogue_domain = ''
            data.extend(parse_dialogue_into_examples(dialogue, dialogue_domain=dialogue_domain, database=database,
                                                     context_len=context_len))

        save_data(data, file_path)

    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    return df


def belief_state_to_str(belief_state: dict[str, dict[str, str]]) -> str:
    result = '{'
    for domain, slot_values in belief_state.items():
        slot_values_str = ','.join(f" {slot} : {value} " for slot, value in slot_values.items() if value != "None")
        if slot_values_str:
            result += f" {domain} {'{'}{slot_values_str}{'}'}"

    result += ' }'
    return result


class MultiWOZDataset:
    def __init__(self,
                 tokenizer_name: str,
                 label_column: str,
                 use_columns: List[str],
                 context_len: int = 1,
                 max_seq_length: int = None,
                 val_size: float = 0.3,
                 additional_special_tokens: List[str] = None,
                 data_path: str = "../huggingface_data",
                 domains: List[str] = None,
                 only_single_domain: bool = False
                 ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.label_column = label_column
        self.use_columns = use_columns
        self.max_seq_length = max_seq_length
        self.context_len = context_len
        self.val_size = val_size
        self.domains = domains
        self.only_single_domain = only_single_domain

        # Initialize pretrained tokenizer and register all the special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True,
                                                       additional_special_tokens=additional_special_tokens)

        print(f"Special tokens: {self.tokenizer.additional_special_tokens}")
        print(f"Domains: {self.domains}")

        # Load train/val/test datasets into DataFrames
        train_df = load_multiwoz_dataset('train', context_len=self.context_len, data_path=data_path,
                                         domains=domains, only_single_domain=self.only_single_domain)
        val_df = load_multiwoz_dataset('validation', context_len=self.context_len, data_path=data_path,
                                       domains=domains, only_single_domain=self.only_single_domain)
        test_df = load_multiwoz_dataset('test', context_len=self.context_len, data_path=data_path,
                                        domains=domains, only_single_domain=self.only_single_domain)

        # Gather unique labels which are used in 'label' <-> 'integers' map
        unique_actions = sorted(list(set([action for example in train_df['actions'].to_list() for action in example])))

        # The 'label' <-> 'integers' map is saved into 'label2id' and 'id2label' dictionaries and saved as a part
        # of the model in model config file.
        self.label2id = {'<UNK_ACT>': 0}
        self.label2id.update({v: k for k, v in enumerate(unique_actions, start=1)})
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)
        print(f"Labels are: \n {self.label2id.keys()}")
        print(f"Number of labels is: {self.num_labels}")

        # Create HuggingFace datasets
        train_dataset = self.create_huggingface_dataset(train_df)
        val_dataset = self.create_huggingface_dataset(val_df)
        test_dataset = self.create_huggingface_dataset(test_df)

        # Create datasets dictionary
        self.dataset = datasets.DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'val': val_dataset}
        )

    def create_huggingface_dataset(self, df: pd.DataFrame, batch_size: int = 32) -> datasets.Dataset:
        """
        Creates HuggingFace dataset from pandas DataFrame
        :param df: input DataFrame
        :param batch_size:
        :return: output HuggingFace dataset
        """
        # Create HuggingFace dataset from Dataset
        dataset = datasets.Dataset.from_pandas(df)

        # Map dataset using the 'tokenize_function'
        dataset = dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)

        dataset = dataset.map(self.cast_labels, batched=True, batch_size=batch_size)

        return dataset

    def tokenize_function(self, example_batch):
        """
        This function prepares each batch for input into the transformer by tokenizing the text, mapping the
        tokenized text into numbers, and adding new arguments with the necessary tensors for input into the model.
        :param example_batch: batch
        :return: augmented batch with added features
        """
        belief_states = [belief_state_to_str(bs) for bs in example_batch['old_belief_state']]
        contexts = list(map(lambda x: self.tokenizer.sep_token.join(x), example_batch['context']))
        texts = [BELIEF + ' ' + belief + ' ' + CONTEXT + ' ' + context + ' ' + USER + ' ' + user_utter
                 for belief, context, user_utter in zip(belief_states, contexts, example_batch['utterance'])]
        example_batch['text'] = texts
        tokenized = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_seq_length)
        return tokenized

    def map_labels_to_ids(self, actions: list[str]) -> list[int]:
        output = [self.label2id.get(a, 0) for a in actions]
        return output

    def cast_labels(self, example_batch):
        labels = np.zeros((len(example_batch['actions']), self.num_labels))
        for idx, action_list in enumerate(example_batch['actions']):
            action_ids = self.map_labels_to_ids(action_list)
            labels[idx, action_ids] = 1.

        return {'label': labels}
