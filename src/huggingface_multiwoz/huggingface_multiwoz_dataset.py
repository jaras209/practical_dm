import logging
import pickle
import random
from abc import abstractmethod
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

logging.basicConfig(level=logging.INFO)


def extract_act_type_slot_name_pairs(dialogue_acts,
                                     strip_domain: bool = False) -> list[str]:
    """
    Extract act_type-slot_name pairs from the dialogue acts in a system turn.

    Args:
        dialogue_acts (dict): A dictionary containing the dialogue acts information from a system turn.
        strip_domain (bool, optional): If True, remove the domain names from the act types. Defaults to False.

    Returns:
        list[str]: A list of act_type-slot_name pairs as strings.
    """
    act_type_slot_name_pairs = []
    act_types = dialogue_acts['dialog_act']['act_type']
    act_slots = dialogue_acts['dialog_act']['act_slots']

    for act_type, act_slot in zip(act_types, act_slots):
        if strip_domain:
            act_type = '-'.join([x for x in act_type.split('-') if x not in DOMAIN_NAMES])

        slot_names = act_slot['slot_name']
        slot_values = act_slot['slot_value']

        for slot_name in slot_names:
            act_type_slot_name_pairs.append(
                f'{act_type}{"-" if slot_name != "none" else ""}{slot_name if slot_name != "none" else ""}'
            )

    return act_type_slot_name_pairs


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
                        frame: Dict[str, List]) -> Dict[str, Dict[str, str]]:
    """
    Updates the belief state based on the given frame.

    Args:
        belief_state (Dict[str, Dict[str, str]]): The current belief state.
        frame (Dict[str, List]): A dictionary containing the domains and states for the current user turn.

    Returns:
        Dict[str, Dict[str, str]]: The updated belief state.
    """

    # Extract the domains and states from the frame
    domains = frame['service']
    states = frame['state']

    # Iterate through each domain and state
    for domain, state in zip(domains, states):
        # Extract the slot names and values from the state
        slots = state['slots_values']['slots_values_name']
        values = state['slots_values']['slots_values_list']

        # Create a dictionary of slot-value pairs
        slot_value_pairs = {slot: value[0] for slot, value in zip(slots, values)}

        # Update the belief state with the new slot-value pairs
        belief_state[domain].update(slot_value_pairs)

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


def parse_dialogue_into_examples(dialogue: Dict[str, Any],
                                 dialogue_domain: str,
                                 database: MultiWOZDatabase,
                                 context_len: Optional[int] = None,
                                 strip_domain: bool = False) -> List[Dict[str, Any]]:
    """
    Parse a dialogue into training examples.

    Args:
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
                'domain': dialogue_domain
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
            act_type_slot_name_pairs = extract_act_type_slot_name_pairs(dialogue_acts, strip_domain=strip_domain)

            context = turns['utterance'][:turn_id - 1]
            if context_len is not None and len(context) > context_len:
                if context_len > 0:
                    context = context[-context_len:]

                else:
                    context = []

            # Update the example dictionary with the new information
            example.update({
                'context': context,
                'actions': list(set(act_type_slot_name_pairs)),
                'system_utterance': utterance,
            })
            examples.append(example)

    return examples


def load_multiwoz_dataset(split: str,
                          database: MultiWOZDatabase,
                          domains: List[str] = None,
                          context_len: int = None,
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
    cache_path = cache_path / ('-'.join(domains) + f"_only-single-domain_{only_single_domain}_context-len_{context_len}")

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
        for dialogue in tqdm(multi_woz_dataset, desc=f"Preprocessing {split} data", unit="dialogue", ncols=100):
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
                                                     context_len=context_len, strip_domain=strip_domain))

        save_data(data, file_path)

    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    return df


def belief_state_to_str(belief_state: Dict[str, Dict[str, str]]) -> str:
    result = '{'
    for domain, slot_values in belief_state.items():
        slot_values_str = ','.join(f" {slot} : {value} " for slot, value in slot_values.items() if value != "None")
        if slot_values_str:
            result += f" {domain} {'{'}{slot_values_str}{'}'}"

    result += ' }'
    return result


def database_results_to_str(database_results: Dict[str, Optional[List[Dict[str, str]]]], threshold: int = 5) -> str:
    result = '{'
    for domain, result_list in database_results.items():
        if result_list is None:
            continue
        elif not result_list:
            result_list_str = "[]"
        else:
            if len(result_list) > threshold:
                result_list = random.sample(result_list, threshold)

            result_list_str = '[' + ','.join(
                '{' + ','.join(f" {slot} : {value} " for slot, value in dict_element.items()) + '}'
                for dict_element in result_list) + ']'

        result += f" {domain} : {result_list_str}"

    result += ' }'
    return result


def database_results_count_to_str(database_results: Dict[str, Optional[List[Dict[str, str]]]]) -> str:
    result = ', '.join(f"{domain} : {len(result_list)}" for domain, result_list in database_results.items()
                       if result_list is not None)
    return '{' + result + '}'


class MultiWOZDataset:
    def __init__(self,
                 context_len: int = 1,
                 root_database_path: Union[str, Path] = "../huggingface_data",
                 domains: List[str] = None,
                 only_single_domain: bool = False):
        """
        Initialize the MultiWOZDataset class.

        Args:
            context_len (int, optional): The maximum length of the conversation history to keep for each example.
            root_database_path (str, Path, optional): The path to the directory where the database is saved.
            domains (List[str], optional): A list of domains to include in the dataset. If None, all domains are included.
            only_single_domain (bool, optional): Whether to include only dialogues with a single domain (if True).

        """
        super().__init__()
        self.context_len = context_len
        self.domains = domains
        self.only_single_domain = only_single_domain

        # Load MultiWoz Database, which is locally saved at database_path
        database_path = Path(root_database_path) / "database"
        self.database = MultiWOZDatabase(database_path)

        logging.info(f"Domains: {self.domains}")

    def load_dataset(self, root_cache_path: Union[str, Path] = "../huggingface_data", strip_domain: bool = False):
        """
        Load the MultiWOZ dataset into a DataFrame.

        Args:
            root_cache_path (str, Path, optional): The path to the directory where the preprocessed cached data will be saved.
            strip_domain (bool, optional): Whether to remove the domain from the action. Defaults to False.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        # Load train/val/test datasets into DataFrames
        train_df = load_multiwoz_dataset('train', database=self.database, context_len=self.context_len,
                                         root_cache_path=root_cache_path, domains=self.domains,
                                         only_single_domain=self.only_single_domain, strip_domain=strip_domain)
        val_df = load_multiwoz_dataset('validation', database=self.database, context_len=self.context_len,
                                       root_cache_path=root_cache_path, domains=self.domains,
                                       only_single_domain=self.only_single_domain, strip_domain=strip_domain)
        test_df = load_multiwoz_dataset('test', database=self.database, context_len=self.context_len,
                                        root_cache_path=root_cache_path, domains=self.domains,
                                        only_single_domain=self.only_single_domain, strip_domain=strip_domain)

        return train_df, val_df, test_df


class MultiWOZDatasetActions:
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
                 strip_domain: bool = False):
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

        """
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.context_len = context_len
        self.domains = domains
        self.only_single_domain = only_single_domain
        self.batch_size = batch_size

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

        logging.info(f"Tokenizer: {self.tokenizer_name}")
        logging.info(f"Special tokens: {self.tokenizer.additional_special_tokens}")
        logging.info(f"Domains: {self.domains}")

        # Gather unique labels which are used in 'label' <-> 'integers' map
        unique_actions = sorted(list(set([action for example in train_df['actions'].to_list() for action in example])))

        # Specify the file path where you want to save the actions
        actions_path = Path(root_cache_path) / "unique_actions.txt"

        # Open the file in write mode
        with actions_path.open("w") as file:
            # Iterate over the elements of the unique_actions list
            for action in unique_actions:
                # Write each action to the file followed by a new line character
                file.write(action + "\n")

        # Initialize LabelEncoder and fit it with the unique labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([UNK_ACTION] + unique_actions)

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
        belief_states = list(map(belief_state_to_str, example_batch['new_belief_state']))

        # Convert the database_results in the example batch into string format and also string with counts
        # database_results = list(map(database_results_to_str, example_batch['database_results']))
        database_results_count = list(map(database_results_count_to_str, example_batch['database_results']))

        # Convert the contexts in the example batch into string format, with elements joined by the separator token.
        contexts = list(map(lambda x: self.tokenizer.sep_token.join(x), example_batch['context']))

        # Combine the belief states, database_results, database_results_count, contexts, and user utterances into
        # a single string for each example in the batch.
        texts = list(map(lambda belief, db_results_count, context, user_utter:
                         BELIEF + ' ' + belief + ' ' + DATABASE_COUNTS + ' ' +
                         db_results_count + ' ' + CONTEXT + ' ' + context + ' ' + USER + ' ' + user_utter,
                         belief_states, database_results_count, contexts, utterances))

        # TODO: try padding to 'longest' instead of 'max_length'
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
            action_ids = self.map_labels_to_ids(action_list)
            # For each action label, set the corresponding cell in the binary array to 1.
            labels[idx, action_ids] = 1.

        # Add tokenized features and labels to the output dictionary.
        tokenized['label'] = labels
        tokenized['text'] = texts

        return tokenized

    def map_labels_to_ids(self, actions: List[str]) -> List[int]:
        """
        Maps action labels to their corresponding integer IDs.

        Args:
            actions (List[str]): A list of action labels.

        Returns:
            List[int]: A list of integer IDs corresponding to the input action labels.
        """
        actions_set = set(actions)
        unseen_actions = actions_set.difference(self.label_encoder.classes_)

        # Map unseen actions to UNK_ACTION
        actions = [UNK_ACTION if action in unseen_actions else action for action in actions]

        # Now we know that all actions are in classes_, so we can call transform safely
        action_ids = self.label_encoder.transform(actions)

        return action_ids.tolist()

    def get_id2label(self) -> Dict[int, str]:
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}

    def get_label2id(self) -> Dict[str, int]:
        return {label: i for i, label in enumerate(self.label_encoder.classes_)}


class MultiWOZBeliefUpdate:
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
                 strip_domain: bool = False):
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

        logging.info(f"Tokenizer: {self.tokenizer_name} with sep_token={self.tokenizer.sep_token}")
        logging.info(f"Special tokens: {self.tokenizer.additional_special_tokens}")
        logging.info(f"Domains: {self.domains}")

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
        utterances = example_batch['utterance']

        old_belief_states = list(map(belief_state_to_str, example_batch['old_belief_state']))
        new_belief_states = list(map(belief_state_to_str, example_batch['new_belief_state']))

        separator = SEPARATOR_BELIEF_UPDATE
        contexts = list(map(lambda x: separator.join(x), example_batch['context']))

        texts = list(map(lambda belief, context, user_utter:
                         TASK_DESCRIPTION_BELIEF_UPDATE + ' ' + ':' + ' ' + BELIEF + ' ' + belief + ' ' + CONTEXT +
                         ' ' + context + ' ' + USER + ' ' + user_utter, old_belief_states, contexts, utterances))

        tokenized_inputs = self.tokenizer(texts, padding='max_length', truncation=True,
                                          max_length=self.max_source_length, return_tensors="pt")

        tokenized_outputs = self.tokenizer(new_belief_states, padding='max_length', truncation=True,
                                           max_length=self.max_target_length, return_tensors="pt")

        # Check if any input texts were truncated
        truncated_texts = [len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_source_length for text in
                           texts]
        num_truncated = sum(truncated_texts)
        if num_truncated > 0:
            logging.warning(f"The number of input truncated texts is: {num_truncated}/ {len(texts)}")

        # Check if any output texts were truncated
        truncated_texts = [len(self.tokenizer(text, truncation=False)['input_ids']) > self.max_target_length
                           for text in new_belief_states]
        num_truncated = sum(truncated_texts)
        if num_truncated > 0:
            logging.warning(f"The number of output truncated texts is: {num_truncated}/ {len(new_belief_states)}")

        # Get labels
        labels = tokenized_outputs['input_ids']

        # Replace padding token id's of the labels by -100, so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': labels,
            'text': texts
        }
