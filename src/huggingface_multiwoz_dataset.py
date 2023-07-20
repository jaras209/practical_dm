import collections
import logging
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import copy
from typing import Optional, List, Any, Tuple, Union, Dict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from database import MultiWOZDatabase
from transformers import AutoTokenizer
from constants import *
from dataframe_utils import print_df_statistics, get_dialogue_subset

logging.basicConfig(level=logging.INFO)

TRUNCATED_INPUTS = 0
TRUNCATED_OUTPUTS = 0


def extract_act_type_slot_name_pairs(dialogue_acts,
                                     strip_domain: bool = False) -> Tuple[List[str], List[Dict]]:
    """
    Extract act_type-slot_name pairs from the dialogue acts in a system turn.

    Args:
        dialogue_acts (dict): A dictionary containing the dialogue acts information from a system turn.
        strip_domain (bool, optional): If True, remove the domain names from the act types. Defaults to False.

    Returns:
        act_type_slot_name_pairs (list[str]): A list of act_type-slot_name pairs as strings.
        string_dialogue_acts (list[str]): A list of dialogue acts constructed from act_type_slot_name_pairs.

    """
    act_type_slot_name_pairs = []
    dict_dialogue_acts = []

    act_types = dialogue_acts['dialog_act']['act_type']
    act_slots = dialogue_acts['dialog_act']['act_slots']

    for act_type, act_slot in zip(act_types, act_slots):
        if strip_domain:
            act_type = '-'.join([x for x in act_type.split('-') if x not in DOMAIN_NAMES])

        slot_names = act_slot['slot_name']
        slot_values = act_slot['slot_value']

        for slot_name, slot_value in zip(slot_names, slot_values):
            if slot_name != 'none':
                act_type_slot_name_pairs.append(f'{act_type}({slot_name})'.lower())
                dict_dialogue_acts.append({act_type.lower(): {slot_name.lower(): slot_value}})
            else:
                act_type_slot_name_pairs.append(act_type.lower())
                dict_dialogue_acts.append({act_type.lower(): None})

    return act_type_slot_name_pairs, dict_dialogue_acts


def get_database_results(database: MultiWOZDatabase,
                         belief_state: Dict[str, Dict[str, str]]) -> Dict[str, Optional[List[Dict[str, str]]]]:
    """
    Get the database results for the current domain and belief state.

    Args:
        database (MultiWOZDatabase): The database instance used for querying.
        belief_state (Dict[str, Dict[str, str]]): The current belief state.

    Returns:
        Dict[str, Optional[List[Dict[str, str]]]]:
            A dictionary with domain as the key and an optional list of resulting database items as the value.
            The list can be empty indicating that no database item matched the given domain state, and the
            value None if the domain was not queried at all.
    """
    database_results = {}

    for domain, domain_state in belief_state.items():
        results = database.query(domain, domain_state)
        database_results[domain] = results

    return database_results


def create_state_update(belief_state: Dict[str, Dict[str, str]],
                        old_belief_state: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Create a state update dictionary representing the difference between the current and old belief states.

    Args:
        belief_state (Dict[str, Dict[str, str]]): The current belief state dictionary.
        old_belief_state (Dict[str, Dict[str, str]]): The old belief state dictionary.

    Returns:
        Dict[str, Dict[str, str]]: The state update dictionary with the same structure as the belief state,
                                   containing the differences between the current and old belief states.
                                   If there's no update for a particular slot, its value will be 'None'.
    """
    state_update = dict()

    for domain, state in belief_state.items():
        state_update[domain] = {}
        for slot, value in state.items():
            # Check if there's an update for the current slot
            if slot not in old_belief_state.get(domain, {}) or old_belief_state[domain][slot] != value:
                state_update[domain][slot] = value
                logging.debug(f"Updating slot '{slot}' in domain '{domain}' with value '{value}'")
            else:
                state_update[domain][slot] = 'None'
                logging.debug(f"No update for slot '{slot}' in domain '{domain}'")

    return state_update


def update_belief_state(belief_state: Dict[str, Dict[str, str]],
                        frame: Dict[str, List],
                        remove_domain: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Updates the belief state based on the given frame.

    Args:
        belief_state (Dict[str, Dict[str, str]]): The current belief state.
        frame (Dict[str, List]): A dictionary containing the domains and states for the current user turn.
        remove_domain (bool, optional): Whether to remove the domain from the slots. Defaults to True.


    Returns:
        Dict[str, Dict[str, str]]: The updated belief state.
    """
    # Extract the domains and states from the frame
    domains = frame['service']
    states = frame['state']

    # Iterate through each domain and state
    for domain, state in zip(domains, states):
        if domain not in belief_state:
            continue

        active_intent = state['active_intent']
        requested_slots = ' '.join(sorted(state['requested_slots']))

        # Extract the slot names and values from the state
        slots = state['slots_values']['slots_values_name']
        values = state['slots_values']['slots_values_list']

        # Create a dictionary of slot-value pairs
        slot_value_pairs = {slot.split('-')[-1] if remove_domain else slot: value[0] for slot, value in
                            zip(slots, values)}

        # Update the belief state with the new slot-value pairs, active intent and requested slots
        belief_state[domain].update(slot_value_pairs)
        belief_state[domain]['intent'] = active_intent
        belief_state[domain]['requested'] = requested_slots if requested_slots else 'None'

    return belief_state


def load_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.

    Args:
        file_path (Path): A pathlib.Path object representing the path to the JSON file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the loaded data.

    Raises:
        FileNotFoundError: If the JSON file is not found at the specified path.
    """

    # Check if the JSON file exists
    if file_path.with_suffix('.json').is_file():
        logging.info(f"Loading data from {file_path.with_suffix('.json')}")

        # Load data from the JSON file
        with open(file_path.with_suffix('.json'), "rb") as f:
            data = pickle.load(f)

        return data
    else:
        raise FileNotFoundError(f"Data file not found at {file_path.with_suffix('.json')}")


def save_data(data: list[[Dict]], file_path: Path, force: bool = False):
    """
        Save data to disk as both JSON and CSV files.

        Args:
            data (list[list[dict]]): A list of lists of dictionaries containing data to be saved.
            file_path (Path): A pathlib.Path object representing the path to save the files to.
            force (bool, optional): Whether to overwrite an existing file with the same path. Defaults to False.

        Raises:
            AssertionError: If the `file_path` already exists and `force` is False.

        Returns:
            None
        """
    if not force:
        assert not file_path.exists(), f"{file_path} already exists."
    with open(file_path.with_suffix('.json'), 'wb') as f:
        pickle.dump(data, f)

    df = pd.DataFrame(data)
    df.to_csv(file_path.with_suffix('.csv'), index=False)


def parse_dialogue_into_examples(dialogue_id: int,
                                 dialogue: Dict[str, Any],
                                 dialogue_domain: str,
                                 database: MultiWOZDatabase,
                                 context_len: Optional[int] = None,
                                 strip_domain: bool = False) -> List[Dict[str, Any]]:
    """
    Parse a dialogue into training examples.

    Args:
        dialogue_id (int): The dialogue ID.
        dialogue (Dict[str, Any]): The dialogue to be parsed.
        dialogue_domain (str): The dialogue domain (e.g., 'restaurant', 'hotel', etc.).
        database (MultiWOZDatabase): The database instance used for querying.
        context_len (Optional[int], optional): The maximum length of the context. Defaults to None.
        strip_domain (bool, optional): Whether to remove the domain from the action. Defaults to False.

    Returns:
        List[Dict[str, Any]]: A list of training examples.
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
                'domain': dialogue_domain,
                'dialogue_id': dialogue_id,
                'user_turn_id': turn_id,
            }

            # Update the belief state based on the user's utterance
            frame = turns['frames'][turn_id]
            belief_state = update_belief_state(belief_state, frame)

            # Create a state update by comparing the old and new belief states
            # state_update = create_state_update(belief_state, example['old_belief_state'])

            # Get the database results for the updated belief state
            database_results = get_database_results(database, belief_state)
            example.update({
                'new_belief_state': copy.deepcopy(belief_state),
                'database_results': database_results
            })

        # SYSTEM
        else:
            dialogue_acts = turns['dialogue_acts'][turn_id]

            # Extract action types and slot names from the dialogue acts
            act_type_slot_name_pairs, string_dialogue_acts = extract_act_type_slot_name_pairs(dialogue_acts,
                                                                                              strip_domain=strip_domain)

            context = turns['utterance'][:turn_id - 1]
            if context_len is not None and len(context) > context_len:
                if context_len > 0:
                    context = context[-context_len:]

                else:
                    context = []

            # Update the example dictionary with the new information
            example.update({
                'context': context,
                'actions': sorted(list(set(act_type_slot_name_pairs))),
                'dialogue_acts': string_dialogue_acts,
                'system_utterance': utterance,
                'system_turn_id': turn_id,
            })
            examples.append(example)

    return examples


def load_multiwoz_dataset(split: str,
                          database: MultiWOZDatabase,
                          domains: List[str] = None,
                          context_len: int = 1,
                          only_single_domain: bool = False,
                          root_cache_path: Union[Path, str] = "data/huggingface_data",
                          strip_domain: bool = False) -> pd.DataFrame:
    """
    Load and preprocess the MultiWOZ 2.2 dataset using the HuggingFace datasets library.

    Args:
        split (str): Which subset of the dataset to load (train, test, or validation).
        database (MultiWOZDatabase): The database with the query method.
        domains (List[str], optional): A list of domains to include in the dataset. If None or empty, all domains are included.
        context_len (int, optional): The maximum length of the conversation history to keep for each example.
        only_single_domain (bool, optional): Whether to include only dialogues with a single domain (if True).
        root_cache_path (Union[Path, str], optional): The path to the directory where the preprocessed cahed data will be saved.
        strip_domain (bool, optional): Whether to remove domain prefixes from slot names in the dataset.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the preprocessed dataset.

    Raises:
        FileNotFoundError: If the MultiWOZDatabase cannot be found at the specified path.
    """
    logging.info(f"Loading MultiWOZ dataset, split={split}, domains={domains}, context_len={context_len}, "
                 f"only_single_domain={only_single_domain}, data_path={root_cache_path}")

    if not domains:
        # If no domains are specified, we use all of them
        domains = list(DOMAIN_NAMES)
        logging.debug(f"Using all domains: {domains}")

    # Sort domains to always get the same order of in the paths and file names below
    domains.sort()

    # Create cache directory path from data path
    cache_path = Path(root_cache_path) / "cache"
    cache_path = cache_path / (
                '-'.join(domains) + f"_only-single-domain_{only_single_domain}_context-len_{context_len}")

    # Create cache directory if it doesn't exist
    cache_path.mkdir(exist_ok=True, parents=True)

    # Create file path for current split (train/test/val)
    file_path = cache_path / f"{split}_preprocessed_data_{'-'.join(domains)}"

    # Turn list of domains into a set to be used in subset queries in filtering dialogues containing only these domains
    domains = set(domains)

    # If the dataset has already been preprocessed, load it from the cache
    if file_path.with_suffix('.json').is_file():
        logging.info(f"Loading {split} from cached file.")
        data = load_data(file_path)

    # Else, load MultiWoz 2.2 dataset from HuggingFace, create data and save them into a cache directory
    else:
        # Load MultiWoz dataset from HuggingFace
        logging.info(f"Preprocessing {split} data and saving it to {file_path}.")
        multi_woz_dataset = datasets.load_dataset(path='multi_woz_v22', split=split, verification_mode='no_checks',
                                                  streaming=True)

        data = []
        # Iterate through dialogues in the dataset, preprocessing each dialogue
        for dialogue_id, dialogue in tqdm(enumerate(multi_woz_dataset), desc=f"Preprocessing {split} data",
                                          unit="dialogue", ncols=100):
            if only_single_domain and len(dialogue['services']) != 1:
                continue

            # Use only those dialogues containing allowed domains and parse them into examples
            if not set(dialogue['services']).issubset(domains):
                continue

            if len(dialogue['services']) > 0:
                dialogue_domain = dialogue['services'][0]
            else:
                dialogue_domain = ''

            data.extend(
                parse_dialogue_into_examples(dialogue_id, dialogue, dialogue_domain=dialogue_domain, database=database,
                                             context_len=context_len, strip_domain=strip_domain))

        save_data(data, file_path)

    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    return df


def belief_state_to_str(belief_state: Dict[str, Dict[str, str]]) -> str:
    result = ''
    for domain, slot_values in sorted(belief_state.items()):
        if slot_values is None:
            continue
        slot_values_str = ', '.join(
            f"{slot}: {value}" for slot, value in sorted(slot_values.items()) if value != "None")
        if slot_values_str:
            result += f"{domain} - {slot_values_str}; "
    return result.rstrip('; ')  # remove the last semicolon and space


def str_to_belief_state(belief_state_str: str, include_none_values: bool = False) -> Dict[str, Dict[str, str]]:
    # Initialize the belief state
    belief_state = {}

    # Split the belief state string into domain sections
    domain_sections = re.split(r';\s*', belief_state_str.strip())
    for domain_section in domain_sections:
        try:
            # Split the domain section into domain name and slots_values_str
            domain_name, slots_values_str = domain_section.split(' - ')

            # Ensure the domain name is recognized
            if domain_name not in DOMAIN_NAMES:
                continue

            # Split slots_values_str into slot-value pairs
            slot_value_pairs = re.split(r',\s*', slots_values_str.strip())
            for slot_value_pair in slot_value_pairs:
                # Split slot_value_pair into slot and value
                slot, value = slot_value_pair.split(': ')

                # Ensure the slot is recognized for this domain
                if slot not in DOMAIN_SLOTS[domain_name]:
                    continue

                # If we're not including "None" values and the value is "None", skip this slot
                if not include_none_values and value.lower() == "none":
                    continue

                # Use setdefault to initialize the domain and slot in the belief state, and update the value
                belief_state.setdefault(domain_name, {}).setdefault(slot, value)

        except ValueError:
            # This occurs if domain_section or slot_value_pair could not be split correctly
            logging.debug(f"Could not parse domain section: {domain_section}")

    # If including "None" values, ensure that all domain-slot pairs exist in the belief state
    if include_none_values:
        for domain, slots in DOMAIN_SLOTS.items():
            for slot in slots:
                belief_state.setdefault(domain, {}).setdefault(slot, "None")

    return belief_state


def database_results_to_str(database_results: Dict[str, Optional[List[Dict[str, str]]]], threshold: int = 5) -> str:
    result = ''
    for domain, result_list in database_results.items():
        if result_list is None:
            continue
        elif not result_list:
            result_list_str = "[]"
        else:
            if len(result_list) > threshold:
                result_list = random.sample(result_list, threshold)

            result_list_str = ', '.join(
                '{' + ','.join(f"{slot}: {value}" for slot, value in dict_element.items()) + '}'
                for dict_element in result_list)

        result += f"{domain} - {result_list_str}; "

    result += '}'
    return result.rstrip('; ')  # remove the last semicolon and space


def database_results_count_to_str(database_results: Dict[str, Optional[List[Dict[str, str]]]]) -> str:
    result = ''
    for domain, result_list in sorted(database_results.items()):
        if result_list is not None:
            result += f"{domain} - {len(result_list)}; "

    return result.rstrip('; ')  # remove the last semicolon and space


def action_list_to_str(action_list: List[str]) -> str:
    # Create a dictionary to group actions by domain
    domain_actions = defaultdict(list)

    # Iterate over each action in the list
    for action in action_list:
        # Split the action into domain and slot (if applicable)
        domain, actions = action.split('-')

        # Append the formatted action to the domain's list
        domain_actions[domain].append(f'{actions}')

    # Generate the final string representation using dictionary and list comprehensions
    string_repr = '; '.join(
        f'{domain} - {", ".join(sorted(actions))}' for domain, actions in sorted(domain_actions.items()))
    return string_repr


def str_to_action_list(action_str: str) -> List[str]:
    # Split the input string into individual domain-action strings
    domain_action_strs = action_str.split(';')

    # Initialize an empty list to hold the actions
    action_list = []

    # Iterate over each domain-action string
    for domain_action_str in domain_action_strs:
        try:
            # Split the domain-action string into domain and action string
            domain, action_str = domain_action_str.split(' - ')

            # Split the action string into individual actions
            actions = action_str.split(', ')

            # Iterate over each action
            for action in actions:
                # Format the action
                formatted_action = f'{domain}-{action}'

                # Check if the action is in the set of all possible actions
                if formatted_action in UNIQUE_ACTIONS:
                    # If it is, add it to the list of actions
                    action_list.append(formatted_action)
        except ValueError:
            # This occurs if domain_action_str could not be split correctly
            logging.debug(f"Could not parse domain-action string: {domain_action_str} in input string {action_str}")

    return action_list


class MultiWOZDatasetActionsClassification:
    def __init__(self,
                 tokenizer_name: str,
                 context_len: int = 1,
                 max_seq_length: int = None,
                 additional_special_tokens: List[str] = None,
                 root_cache_path: Union[str, Path] = "../huggingface_data",
                 root_database_path: Union[str, Path] = "../huggingface_data",
                 domains: List[str] = None,
                 only_single_domain: bool = False,
                 batch_size: int = 32,
                 strip_domain: bool = False,
                 min_action_support: int = 5,
                 subset_size: float = 0.3):
        """
        Initialize the MultiWOZDataset class.

        Args:
            tokenizer_name (str): The name of the tokenizer to use.
            context_len (int, optional): The maximum length of the conversation history to keep for each example.
            max_seq_length (int, optional): The maximum sequence length for tokenized input.
            additional_special_tokens (List[str], optional): A list of additional special tokens to use with the tokenizer.
            root_cache_path (str, Path, optional): The path to the directory where the preprocessed cached data will be saved.
            root_database_path (str, Path, optional): The path to the directory where the database is saved.
            domains (List[str], optional): A list of domains to include in the dataset. If None, all domains are included.
            only_single_domain (bool, optional): Whether to include only dialogues with a single domain (if True).
            batch_size (int, optional): The batch size to use when processing the dataset.
            strip_domain (bool, optional): Whether to remove the domain from the action. Defaults to False.
            min_action_support (int, optional): Set the minimum action support size to consider that action in train data

        """
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.context_len = context_len
        self.domains = domains
        self.only_single_domain = only_single_domain
        self.batch_size = batch_size
        self.min_action_support = min_action_support  # set min action support

        # Initialize pretrained tokenizer and register all the special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True,
                                                       additional_special_tokens=additional_special_tokens)

        # Load MultiWoz Database, which is locally saved at database_path
        database_path = Path(root_database_path) / "database"
        self.database = MultiWOZDatabase(database_path)

        # Load train/val/test datasets into DataFrames
        train_df = load_multiwoz_dataset('train', database=self.database, context_len=self.context_len,
                                         root_cache_path=root_cache_path, domains=domains,
                                         only_single_domain=self.only_single_domain, strip_domain=strip_domain)
        val_df = load_multiwoz_dataset('validation', database=self.database, context_len=self.context_len,
                                       root_cache_path=root_cache_path, domains=domains,
                                       only_single_domain=self.only_single_domain, strip_domain=strip_domain)
        test_df = load_multiwoz_dataset('test', database=self.database, context_len=self.context_len,
                                        root_cache_path=root_cache_path, domains=domains,
                                        only_single_domain=self.only_single_domain, strip_domain=strip_domain)

        old_train_df = train_df

        # Get the subset dataframe
        train_df = get_dialogue_subset(train_df, subset_size)

        # Print statistics
        print_df_statistics(df=old_train_df, df_subset=train_df)

        logging.info(f"Tokenizer: {self.tokenizer_name}")
        logging.info(f"Special tokens: {self.tokenizer.additional_special_tokens}")
        logging.info(f"Domains: {self.domains}")

        # Create actions, count their support and filter the actions that have low support.
        #   Count the occurrence of each action
        action_counts = collections.Counter([action for example in train_df['actions'].to_list() for action in example])

        #   Create a DataFrame from the action_counts dictionary
        actions_df = pd.DataFrame(list(action_counts.items()), columns=['action', 'support'])
        actions_df['supported'] = actions_df['support'] >= self.min_action_support
        actions_df = actions_df.sort_values(by='support', ascending=False)

        #   Specify the file path and save the DataFrame as CSV
        # actions_path = Path(root_cache_path) / "action_supports.csv"
        # actions_df.to_csv(actions_path, index=False)

        #   Filter out actions with support less than the threshold
        supported_actions = actions_df[actions_df['supported']]['action'].tolist()

        # Initialize LabelEncoder and fit it with the supported actions
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([UNK_ACTION] + supported_actions)

        # Get the number of unique labels
        self.num_labels = len(self.label_encoder.classes_)
        logging.debug(f"Labels are: \n {self.label_encoder.classes_}")
        logging.info(f"Number of labels is: {self.num_labels}")

        # Create HuggingFace datasets
        train_dataset = self.create_huggingface_dataset(train_df)
        val_dataset = self.create_huggingface_dataset(val_df)
        test_dataset = self.create_huggingface_dataset(test_df)

        # Create dataset dictionary
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
        # Create HuggingFace dataset from Dataset
        dataset = datasets.Dataset.from_pandas(df)

        # Map dataset using the 'tokenize_and_cast_function'
        dataset = dataset.map(self.tokenize_and_cast_function, batched=True, batch_size=self.batch_size)

        return dataset

    def tokenize_and_cast_function(self,
                                   example_batch: Dict[str, Any]) -> Dict[str, Union[List[str], List[int], np.ndarray]]:
        """
        Tokenizes the input text and casts string labels (action names) into boolean vectors.

        Args:
            example_batch (Dict[str, Any]): A dictionary where each key is a column name and each value corresponds
                to the values of that column for all examples in the batch.

        Returns: Dict[str, Union[List[str], List[int], np.ndarray]]: A dictionary where each key is a feature name
         and each value corresponds to the values of that feature for all examples in the batch. The returned
         dictionary includes tokenized features ('input_ids', 'token_type_ids', 'attention_mask') and labels in binary
         array format.
        """
        utterances = example_batch['utterance']

        # Convert the belief states in the example batch into string format.
        new_belief_states = list(map(belief_state_to_str, example_batch['new_belief_state']))

        # Convert the database_results in the example batch into string format and also string with counts
        # database_results = list(map(database_results_to_str, example_batch['database_results']))
        database_results_count = list(map(database_results_count_to_str, example_batch['database_results']))

        # Convert the contexts in the example batch into string format, with elements joined by the separator token.
        contexts = list(map(lambda x: self.tokenizer.sep_token.join(x), example_batch['context']))

        # Combine the belief states, database_results, database_results_count, contexts, and user utterances into
        # a single string for each example in the batch.
        texts = list(map(lambda belief, context, user_utter, db_results_count:
                         STATE_CLA + ' ' + belief + '. '
                         + CONTEXT_CLA + ' ' + context + '. ' + USER + ' ' + user_utter + '. ' +
                         DATABASE_COUNTS + ' ' + db_results_count,
                         new_belief_states, contexts, utterances, database_results_count))

        # Use the tokenizer to convert these strings into a format suitable for model input
        tokenized = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_seq_length,
                                   return_tensors='pt')

        # Check if any input texts were truncated
        truncated_texts = [len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_seq_length for text in
                           texts]
        num_truncated = sum(truncated_texts)
        if num_truncated > 0:
            logging.warning(f"The number of input truncated texts is: {num_truncated}/ {len(texts)}")

        # Cast action labels to binary arrays.
        # Create a binary array where each row corresponds to an example,
        # and each column corresponds to a unique action.
        labels = np.zeros((len(example_batch['actions']), self.num_labels))
        for idx, action_list in enumerate(example_batch['actions']):
            # Convert the action labels into their corresponding indices.
            action_ids = self.map_labels_to_ids(action_list, filter_unseen=True)
            # For each action label, set the corresponding cell in the binary array to 1.
            labels[idx, action_ids] = 1.

        # Add tokenized features and labels to the output dictionary.
        tokenized['label'] = labels
        tokenized['text'] = texts

        return tokenized

    def map_labels_to_ids(self, actions: List[str], filter_unseen: bool = False) -> List[int]:
        """
        Maps action labels to their corresponding integer IDs.

        Args:
            actions (List[str]): A list of action labels.
            filter_unseen (bool): If True, unseen actions are filtered out. If False, unseen actions are replaced with UNK_ACTION.

        Returns:
            List[int]: A list of integer IDs corresponding to the input action labels.
        """
        actions_set = set(actions)
        unseen_actions = actions_set.difference(self.label_encoder.classes_)

        if filter_unseen:
            # Filter out unseen actions
            actions = [action for action in actions if action not in unseen_actions]
        else:
            # Replace unseen actions with UNK_ACTION
            actions = [UNK_ACTION if action in unseen_actions else action for action in actions]

        # Now we know that all actions are in classes_, so we can call transform safely
        action_ids = self.label_encoder.transform(actions)

        return action_ids.tolist()

    def get_id2label(self) -> Dict[int, str]:
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}

    def get_label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.label_encoder.classes_)}


class MultiWOZDatasetBeliefUpdate:
    def __init__(self,
                 tokenizer_name: str,
                 context_len: int = 1,
                 max_source_length: int = None,
                 max_target_length: int = None,
                 additional_special_tokens: List[str] = None,
                 root_cache_path: Union[str, Path] = "../huggingface_data",
                 root_database_path: Union[str, Path] = "../huggingface_data",
                 domains: List[str] = None,
                 only_single_domain: bool = False,
                 batch_size: int = 32,
                 strip_domain: bool = False,
                 subset_size: float = 0.3):
        """
        Initialize the MultiWOZDataset class.

        Args:
            tokenizer_name (str): The name of the tokenizer to use.
            context_len (int, optional): The maximum length of the conversation history to keep for each example.
            max_source_length (int, optional): The maximum sequence length for tokenized input.
            max_target_length (int, optional): The maximum sequence length for the output.
            additional_special_tokens (List[str], optional): A list of additional special tokens to use with the tokenizer.
            root_cache_path (str, Path, optional): The path to the directory where the preprocessed cached data will be saved.
            root_database_path (str, Path, optional): The path to the directory where the database is saved.
            domains (List[str], optional): A list of domains to include in the dataset. If None, all domains are included.
            only_single_domain (bool, optional): Whether to include only dialogues with a single domain (if True).
            batch_size (int, optional): The batch size to use when processing the dataset.
            strip_domain (bool, optional): Whether to remove the domain from the action. Defaults to False.

        """
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.context_len = context_len
        self.domains = domains
        self.only_single_domain = only_single_domain
        self.batch_size = batch_size

        # Initialize pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        # Load MultiWoz Database, which is locally saved at database_path
        database_path = Path(root_database_path) / "database"
        self.database = MultiWOZDatabase(database_path)

        # Load train/val/test datasets into DataFrames
        train_df = load_multiwoz_dataset('train', database=self.database, context_len=self.context_len,
                                         root_cache_path=root_cache_path, domains=domains,
                                         only_single_domain=self.only_single_domain, strip_domain=strip_domain)
        val_df = load_multiwoz_dataset('validation', database=self.database, context_len=self.context_len,
                                       root_cache_path=root_cache_path, domains=domains,
                                       only_single_domain=self.only_single_domain, strip_domain=strip_domain)
        test_df = load_multiwoz_dataset('test', database=self.database, context_len=self.context_len,
                                        root_cache_path=root_cache_path, domains=domains,
                                        only_single_domain=self.only_single_domain, strip_domain=strip_domain)

        old_train_df = train_df

        # Get the subset dataframe
        train_df = get_dialogue_subset(train_df, subset_size)

        # Print statistics
        print_df_statistics(df=old_train_df, df_subset=train_df)

        logging.info(f"Tokenizer: {self.tokenizer_name} with sep_token={self.tokenizer.sep_token}")
        logging.info(f"Domains: {self.domains}")

        # Create HuggingFace datasets
        train_dataset = self.create_huggingface_dataset(train_df)
        val_dataset = self.create_huggingface_dataset(val_df)
        test_dataset = self.create_huggingface_dataset(test_df)

        # Log number of truncated input and output examples
        logging.info(f"Truncated inputs: {TRUNCATED_INPUTS}, truncated outputs: {TRUNCATED_OUTPUTS}")

        # Create dataset dictionary
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
        # Create HuggingFace dataset from Dataset
        dataset = datasets.Dataset.from_pandas(df)

        # Map dataset using the 'tokenize_and_cast_function'
        dataset = dataset.map(self.tokenize_and_cast_function, batched=True, batch_size=self.batch_size)

        return dataset

    def tokenize_and_cast_function(self,
                                   example_batch: Dict[str, Any]) -> Dict[str, Union[List[str], List[int], np.ndarray]]:
        global TRUNCATED_INPUTS, TRUNCATED_OUTPUTS

        utterances = example_batch['utterance']

        old_belief_states = list(map(belief_state_to_str, example_batch['old_belief_state']))
        new_belief_states = list(map(belief_state_to_str, example_batch['new_belief_state']))

        separator = CONTEXT_SEP
        contexts = list(map(lambda x: separator.join(x), example_batch['context']))

        texts = list(map(lambda belief, context, user_utter:
                         TASK_DESCRIPTION_STATE_UPDATE + ' ' + STATE_GEN + ' ' + belief + '. '
                         + CONTEXT_GEN + ' ' + context + '. ' + USER_GEN + ' ' + user_utter,
                         old_belief_states, contexts, utterances))

        tokenized_inputs = self.tokenizer(texts, padding='max_length', truncation=True,
                                          max_length=self.max_source_length, return_tensors="pt")

        tokenized_outputs = self.tokenizer(new_belief_states, padding='max_length', truncation=True,
                                           max_length=self.max_target_length, return_tensors="pt")

        # Check if any input texts were truncated
        truncated_texts = [text for text in texts if
                           len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_source_length]
        num_truncated = len(truncated_texts)
        if num_truncated > 0:
            logging.warning(f"The number of input truncated texts is: {num_truncated}/ {len(texts)}")
            TRUNCATED_INPUTS += num_truncated

        # Check if any output texts were truncated
        truncated_texts = [text for text in new_belief_states if
                           len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_target_length]
        num_truncated = len(truncated_texts)
        if num_truncated > 0:
            logging.warning(f"The number of output truncated texts is: {num_truncated}/ {len(texts)}")
            TRUNCATED_OUTPUTS += num_truncated

        # Get labels
        labels = tokenized_outputs['input_ids']

        # Replace padding token id's of the labels by -100, so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': labels,
        }


class MultiWOZDatasetActionGeneration:
    def __init__(self,
                 tokenizer_name: str,
                 context_len: int = 1,
                 max_source_length: int = None,
                 max_target_length: int = None,
                 root_cache_path: Union[str, Path] = "../huggingface_data",
                 root_database_path: Union[str, Path] = "../huggingface_data",
                 domains: List[str] = None,
                 only_single_domain: bool = False,
                 batch_size: int = 32,
                 strip_domain: bool = False,
                 min_action_support: int = 10,
                 subset_size: float = 0.3):
        """
        Initialize the MultiWOZDataset class.

        Args:
            tokenizer_name (str): The name of the tokenizer to use.
            context_len (int, optional): The maximum length of the conversation history to keep for each example.
            max_source_length (int, optional): The maximum sequence length for tokenized input.
            max_target_length (int, optional): The maximum sequence length for the output.
            root_cache_path (str, Path, optional): The path to the directory where the preprocessed cached data will be saved.
            root_database_path (str, Path, optional): The path to the directory where the database is saved.
            domains (List[str], optional): A list of domains to include in the dataset. If None, all domains are included.
            only_single_domain (bool, optional): Whether to include only dialogues with a single domain (if True).
            batch_size (int, optional): The batch size to use when processing the dataset.
            strip_domain (bool, optional): Whether to remove the domain from the action. Defaults to False.

        """
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.context_len = context_len
        self.domains = domains
        self.only_single_domain = only_single_domain
        self.batch_size = batch_size
        self.min_action_support = min_action_support

        # Initialize pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        # Load MultiWoz Database, which is locally saved at database_path
        database_path = Path(root_database_path) / "database"
        self.database = MultiWOZDatabase(database_path)

        # Load train/val/test datasets into DataFrames
        train_df = load_multiwoz_dataset('train', database=self.database, context_len=self.context_len,
                                         root_cache_path=root_cache_path, domains=domains,
                                         only_single_domain=self.only_single_domain, strip_domain=strip_domain)
        val_df = load_multiwoz_dataset('validation', database=self.database, context_len=self.context_len,
                                       root_cache_path=root_cache_path, domains=domains,
                                       only_single_domain=self.only_single_domain, strip_domain=strip_domain)
        test_df = load_multiwoz_dataset('test', database=self.database, context_len=self.context_len,
                                        root_cache_path=root_cache_path, domains=domains,
                                        only_single_domain=self.only_single_domain, strip_domain=strip_domain)

        old_train_df = train_df

        # Get the subset dataframe
        train_df = get_dialogue_subset(train_df, subset_size)

        # Print statistics
        print_df_statistics(df=old_train_df, df_subset=train_df)

        logging.info(f"Tokenizer: {self.tokenizer_name} with sep_token={self.tokenizer.sep_token}")
        logging.info(f"Domains: {self.domains}")

        # Create actions, count their support and filter the actions that have low support.
        #   Count the occurrence of each action
        action_counts = collections.Counter([action for example in train_df['actions'].to_list() for action in example])

        #   Create a DataFrame from the action_counts dictionary
        actions_df = pd.DataFrame(list(action_counts.items()), columns=['action', 'support'])
        actions_df['supported'] = actions_df['support'] >= self.min_action_support
        actions_df = actions_df.sort_values(by='support', ascending=False)

        #   Specify the file path and save the DataFrame as CSV
        # actions_path = Path(root_cache_path) / "action_supports.csv"
        # actions_df.to_csv(actions_path, index=False)

        #   Filter out actions with support less than the threshold
        self.supported_actions = set(actions_df[actions_df['supported']]['action'].tolist())

        # Create HuggingFace datasets
        train_dataset = self.create_huggingface_dataset(train_df)
        val_dataset = self.create_huggingface_dataset(val_df)
        test_dataset = self.create_huggingface_dataset(test_df)

        # Log number of truncated input and output examples
        logging.info(f"Truncated inputs: {TRUNCATED_INPUTS}, truncated outputs: {TRUNCATED_OUTPUTS}")

        # Create dataset dictionary
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
        # Create HuggingFace dataset from Dataset
        dataset = datasets.Dataset.from_pandas(df)

        # Map dataset using the 'tokenize_and_cast_function'
        dataset = dataset.map(self.tokenize_and_cast_function, batched=True, batch_size=self.batch_size)

        return dataset

    def tokenize_and_cast_function(self, example_batch: Dict[str, Any]) -> Dict[
        str, Union[List[str], List[int], np.ndarray]]:
        global TRUNCATED_INPUTS, TRUNCATED_OUTPUTS

        utterances = example_batch['utterance']

        # Convert the belief states in the example batch into string format.
        new_belief_states = list(map(belief_state_to_str, example_batch['new_belief_state']))

        # Filter only those actions that are supported
        filtered_actions = [[a for a in action if a in self.supported_actions] for action in example_batch['actions']]

        # Convert action lists into a string format
        actions = list(map(action_list_to_str, filtered_actions))

        # Convert the database_results in the example batch into string format with counts
        database_results_count = list(map(database_results_count_to_str, example_batch['database_results']))

        # Convert the contexts in the example batch into string format, with elements joined by the separator token.
        separator = CONTEXT_SEP
        contexts = list(map(lambda x: separator.join(x), example_batch['context']))

        texts = list(map(lambda belief, context, user_utter, db_results_count:
                         TASK_DESCRIPTION_ACTION_GENERATION + ' ' + STATE_GEN + ' ' + belief + '. '
                         + CONTEXT_GEN + ' ' + context + '. ' + USER_GEN + ' ' + user_utter + '. ' +
                         DATABASE_COUNTS_GEN + ' ' + db_results_count,
                         new_belief_states, contexts, utterances, database_results_count))

        tokenized_inputs = self.tokenizer(texts, padding='max_length', truncation=True,
                                          max_length=self.max_source_length, return_tensors="pt")

        tokenized_outputs = self.tokenizer(actions, padding='max_length', truncation=True,
                                           max_length=self.max_target_length, return_tensors="pt")

        # Check if any input texts were truncated
        truncated_texts = [text for text in texts if
                           len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_source_length]
        num_truncated = len(truncated_texts)
        if num_truncated > 0:
            logging.warning(f"The number of input truncated texts is: {num_truncated}/ {len(texts)}")
            TRUNCATED_INPUTS += num_truncated

        # Check if any output texts were truncated
        truncated_texts = [text for text in actions if
                           len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_target_length]
        num_truncated = len(truncated_texts)
        if num_truncated > 0:
            logging.warning(f"The number of output truncated texts is: {num_truncated}/ {len(texts)}")
            TRUNCATED_OUTPUTS += num_truncated

        # Get labels
        labels = tokenized_outputs['input_ids']

        # Replace padding token id's of the labels by -100, so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': labels,
            'new_belief_state': new_belief_states,
        }
