import json

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset

DOMAIN_NAMES = ['Restaurant', 'Attraction', 'Hotel', 'Taxi', 'Train', 'Bus', 'Hospital', 'Police']

# TODO: `dataset_dir` must be changed to '/home/safar/HCN/data' when using AIC. Locally, use '../data'.
# DATASET_DIR = '/home/safar/HCN/data'
DATASET_DIR = '../data'


def load_data(dataset_type: str, k: int, dataset_dir: str = DATASET_DIR, domains: list = None,
              strip_domain: bool = True) -> pd.DataFrame:
    """
    Loads data from the specified directory.

    :param dataset_dir: dataset directory
    :param dataset_type: specify dataset type - `train`, `validation` or `test`
    :param k:
    :param domains:
    :param strip_domain:
    :return: list of dataset examples, where each example is a dictionary of the following structure:
        `{  'user_utterance': str,       # the string with the current user response
            'system_utterance': str,     # the string with the current system response
            'system_actions': list[str]  # list of system action names for the current turn
            'context': list[str],        # list of utterances preceding the current utterance
        }`
        # TODO: change this docstring to correspond to reality, where we return pd.DataFrame. Also column dialogue.
    """
    assert dataset_type in ['train', 'val', 'test'], \
        AssertionError(f'Wrong dataset_type \'{dataset_type}\'! Only acceptable dataset_type is \'train\', '
                       f'\'val\' or \'test\'.')

    dataset_path = Path(dataset_dir) / dataset_type
    acts_path = Path(dataset_dir) / 'dialog_acts.json'
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
                        'dialogue': context.copy() + [example['user_utterance']]
                    })
                    data.append(example)
                    context.append(example['user_utterance'])
                    context.append(example['system_utterance'])
                    if len(context) > k and len(context) >= 2:
                        context = context[1:]

    return pd.DataFrame(data)


def create_action_map(df: pd.DataFrame):
    unique_actions = sorted(list(set([action for example in df.system_actions for action in example])))
    action_to_ids = {'<UNK_ACT>': 0}
    action_to_ids.update({v: k for k, v in enumerate(unique_actions, start=1)})
    ids_to_action = {v: k for k, v in action_to_ids.items()}
    return action_to_ids, ids_to_action


def convert_actions_to_ids(actions: list[str], dictionary: dict) -> list[int]:
    output = [dictionary.get(a, 0) for a in actions]
    return output


def create_dataset(df: pd.DataFrame, action_to_ids: dict, tokenizer) -> Dataset:
    """
    Builds

    """
    def encode(examples):
        text = list(map(lambda x: tokenizer.sep_token.join(x), examples['dialogue']))
        examples['text'] = text

        system_actions = np.zeros((len(examples['text']), len(action_to_ids)), dtype=np.float)
        for i, action_list in enumerate(examples['system_actions']):
            action_ids = convert_actions_to_ids(action_list, action_to_ids)
            system_actions[i, action_ids] = 1

        examples['label'] = system_actions
        return tokenizer(examples['text'], max_length=512, padding="max_length", truncation=True)

    dataset = Dataset.from_pandas(df).map(encode, batched=True)
    return dataset

