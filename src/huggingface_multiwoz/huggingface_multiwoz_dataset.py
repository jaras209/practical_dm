import logging
import pickle
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import copy
from typing import Optional, List, Any, Tuple, Union, Dict
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from database import MultiWOZDatabase
from transformers import AutoTokenizer
from constants import *
from huggingface_multiwoz.dialogue_processing import parse_dialogue_into_examples

logging.basicConfig(level=logging.INFO)


def _compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = (logits >= 0).astype(np.float32)
    return {"accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "recall": recall_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "precision": precision_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "f1": f1_score(y_true=y_true, y_pred=y_pred, average='weighted')}


def save_data(data: List[Dict[str, Any]], file_path: Path, append: bool = False) -> None:
    """
    Save data to disk as both JSON and CSV files.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries containing data to be saved.
        file_path (Path): A pathlib.Path object representing the path to save the files to.
        append (bool, optional): Whether to append data to an existing file or overwrite the file. Defaults to False.
    """

    # If append is True and the file already exists, load the existing data and extend it with the new data
    if append and file_path.with_suffix('.json').is_file():
        with open(file_path.with_suffix('.json'), 'rb') as f:
            existing_data = pickle.load(f)
        data = existing_data + data

    # Save the (extended) data as a JSON file using 'wb' mode
    with open(file_path.with_suffix('.json'), 'wb') as f:
        pickle.dump(data, f)

    # Convert the data to a DataFrame and save it as a CSV file
    df = pd.DataFrame(data)
    write_header = not append or not file_path.with_suffix('.csv').is_file()
    df.to_csv(file_path.with_suffix('.csv'), index=False, mode=('a' if append else 'w'), header=write_header)

    logging.debug(f"Data saved to {file_path.with_suffix('.json')} and {file_path.with_suffix('.csv')}")


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


def load_multiwoz_dataset(split: str,
                          domains: List[str] = None,
                          context_len: Optional[int] = None,
                          only_single_domain: bool = False,
                          data_path: Union[Path, str] = "data/huggingface_data",
                          strip_domain: bool = False,
                          save_interval: int = 1000) -> pd.DataFrame:
    """
    Load and preprocess the MultiWOZ 2.2 dataset using the HuggingFace datasets library.

    Args:
        split (str): Which subset of the dataset to load (train, test, or validation).
        domains (List[str], optional): A list of domains to include in the dataset. If None, all domains are included.
        context_len (int, optional): The maximum length of the conversation history to keep for each example.
        only_single_domain (bool, optional): Whether to include only dialogues with a single domain (if True).
        data_path (Union[Path, str], optional): The path to the directory where the preprocessed data should be saved.
        strip_domain (bool, optional): Whether to remove domain prefixes from slot names in the dataset.
        save_interval (int, optional): The number of dialogues to process before saving the data to a file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the preprocessed dataset.

    Raises:
        FileNotFoundError: If the MultiWOZDatabase cannot be found at the specified path.
    """

    logging.info(f"Loading MultiWOZ dataset, split={split}, domains={domains}, context_len={context_len}, "
                 f"only_single_domain={only_single_domain}, data_path={data_path}")

    if not domains:
        # If no domains are specified, we use all of them
        domains = list(DOMAIN_NAMES)
        logging.debug(f"Using all domains: {domains}")

    # Sort domains to always get the same order of in the paths and file names below
    domains.sort()

    # Create cache directory path and database directory path from data path
    cache_path = Path(data_path) / "cache"
    database_path = Path(data_path) / "database"

    cache_path = cache_path / ('-'.join(domains) + f"_only-single-domain_{only_single_domain}")

    # Create cache directory, if it doesn't exist
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
        multi_woz_dataset = datasets.load_dataset(path='multi_woz_v22', split=split, ignore_verifications=True,
                                                  streaming=True)

        # Load MultiWoz Database, which is locally saved at database_path
        database = MultiWOZDatabase(database_path)

        batch_data = []
        dialogue_counter = 0

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

            processed_dialogue = parse_dialogue_into_examples(dialogue, dialogue_domain=dialogue_domain,
                                                              database=database, context_len=context_len,
                                                              strip_domain=strip_domain)
            if processed_dialogue:
                batch_data.extend(processed_dialogue)
                dialogue_counter += 1

                if dialogue_counter % save_interval == 0:
                    save_data(batch_data, file_path, append=True)
                    batch_data = []

            # Save remaining dialogues
            if batch_data:
                save_data(batch_data, file_path, append=True)

        # Load data from saved files to create DataFrame
        data = load_data(file_path)

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
                 only_single_domain: bool = False,
                 batch_size: int = 32):
        """
        Initialize the MultiWOZDataset class.

        Args:
            tokenizer_name (str): The name of the tokenizer to use.
            label_column (str): The name of the label column in the dataset.
            use_columns (List[str]): A list of columns to use in the dataset.
            context_len (int, optional): The maximum length of the conversation history to keep for each example.
            max_seq_length (int, optional): The maximum sequence length for tokenized input.
            val_size (float, optional): The proportion of the dataset to use for validation.
            additional_special_tokens (List[str], optional): A list of additional special tokens to use with the tokenizer.
            data_path (str, optional): The path to the directory where the preprocessed data should be saved.
            domains (List[str], optional): A list of domains to include in the dataset. If None, all domains are included.
            only_single_domain (bool, optional): Whether to include only dialogues with a single domain (if True).
            batch_size (int, optional): The batch size to use when processing the dataset.
        """
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.label_column = label_column
        self.use_columns = use_columns
        self.max_seq_length = max_seq_length
        self.context_len = context_len
        self.val_size = val_size
        self.domains = domains
        self.only_single_domain = only_single_domain
        self.batch_size = batch_size

        # Initialize pretrained tokenizer and register all the special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True,
                                                       additional_special_tokens=additional_special_tokens)

        logging.info(f"Special tokens: {self.tokenizer.additional_special_tokens}")
        logging.info(f"Domains: {self.domains}")

        # Load train/val/test datasets into DataFrames
        train_df = load_multiwoz_dataset('train', context_len=self.context_len, data_path=data_path,
                                         domains=domains, only_single_domain=self.only_single_domain)
        val_df = load_multiwoz_dataset('validation', context_len=self.context_len, data_path=data_path,
                                       domains=domains, only_single_domain=self.only_single_domain)
        test_df = load_multiwoz_dataset('test', context_len=self.context_len, data_path=data_path,
                                        domains=domains, only_single_domain=self.only_single_domain)

        # Gather unique labels which are used in 'label' <-> 'integers' map
        unique_actions = sorted(list(set([action for example in train_df['actions'].to_list() for action in example])))

        # Initialize LabelEncoder and fit it with the unique labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['<UNK_ACT>'] + unique_actions)

        # Get the number of unique labels
        self.num_labels = len(self.label_encoder.classes_)
        logging.debug(f"Labels are: \n {self.label_encoder.classes_}")
        logging.info(f"Number of labels is: {self.num_labels}")

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

        # Map dataset using the 'tokenize_function'
        dataset = dataset.map(self.tokenize_function, batched=True, batch_size=self.batch_size)

        dataset = dataset.map(self.cast_labels, batched=True, batch_size=self.batch_size)

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

    def map_labels_to_ids(self, actions: List[str]) -> List[int]:
        output = self.label_encoder.transform(actions)
        return output

    def get_id2label(self) -> dict[int, str]:
        return {self.label_encoder.transform(label): label for label in self.label_encoder.classes_}

    def get_label2id(self) -> dict[str, int]:
        return {label: self.label_encoder.transform(label) for label in self.label_encoder.classes_}

    def cast_labels(self, example_batch):
        labels = np.zeros((len(example_batch['actions']), self.num_labels))
        for idx, action_list in enumerate(example_batch['actions']):
            action_ids = self.map_labels_to_ids(action_list)
            labels[idx, action_ids] = 1.

        return {'label': labels}
