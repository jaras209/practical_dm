import json
import pickle
import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
import torch.utils.data as torchdata
import copy
from tqdm import tqdm
from typing import Optional, List, Any, Tuple
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from transformers import (
    AutoTokenizer,
    AddedToken,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    pipelines
)
from transformers.pipelines.pt_utils import KeyDataset

BELIEF = '<|BELIEF|>'
CONTEXT = '<|CONTEXT|>'
USER = '<|USER|>'
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
parser.add_argument("--pretrained_model",
                    # default='/home/safar/HCN/models/hf_multiwoz_hotel/roberta-base/roberta-base_230315-170036_230316-124221',
                    default='roberta-base',
                    # default='../models/hf_hotel_roberta-base_230316-151206',
                    type=str,
                    help="Path to the pretrained Hugging face model.")
parser.add_argument("--tokenizer_name", default='roberta-base', type=str,
                    help="Path to the pretrained Hugging face tokenizer.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--max_seq_length", default=None, type=int, help="Max seq length of input to transformer")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--early_stopping_patience", default=10, type=int, help="Number of epochs after which the "
                                                                            "training is ended if there is no "
                                                                            "improvement on validation data")
parser.add_argument("--save_folder", default="/home/safar/HCN/models/hf_multiwoz_train", type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument("--cache_dir",
                    default="/home/safar/HCN/huggingface_dataset_train",
                    # default="../huggingface_dataset_hotel",
                    type=str,
                    help="Name of the folder where to save extracted multiwoz dataset for faster preprocessing.")
parser.add_argument("--domains", default=['train'], nargs='*')
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
args = parser.parse_args([] if "__file__" not in globals() else None)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = (logits >= 0).astype(np.float32)
    return {"accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
            "recall": recall_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "precision": precision_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            "f1": f1_score(y_true=y_true, y_pred=y_pred, average='weighted')}


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
            # Create example instance with user utterance and old belief state, i.e. belief state, which is in the
            # input to our model. It represents the belief state of our system after the previous turn and will be on
            # the output in the previous turn.
            example = {
                'utterance': utterance,
                'old_belief_state': copy.deepcopy(belief_state),
            }
            frame = turns['frames'][turn_id]
            domains = frame['service']
            states = frame['state']

            for domain, state in zip(domains, states):
                slots = state['slots_values']['slots_values_name']
                values = state['slots_values']['slots_values_list']

                slot_value_pairs = {slot: value[0] for slot, value in zip(slots, values)}
                belief_state[domain].update(slot_value_pairs)

            # From the USER we use:
            #   - 'utterance': what the user said in the current turn
            #   - 'belief_state': the belief state of the user side of the conversation
            example.update({'new_belief_state': copy.deepcopy(belief_state)})

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


def load_multiwoz_dataset(split: str, domains: List[str] = None, context_len=None,
                          cache_dir="/home/safar/HCN/huggingface_dataset") -> pd.DataFrame:
    """
    Load the MultiWoz dataset using huggingface.datasets.
    Able to shorten the context length by setting context_len.
    """

    cache_dir = Path(cache_dir)
    print(f"Cache dir = {Path(cache_dir).absolute()}")

    # Create cache dir if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if domains is None:
        domains = []
    domains.sort()

    f_name = os.path.join(cache_dir, f"{split}_preprocessed_data_{'-'.join(domains)}")

    domains = set(domains)

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

            if not domains or set(dialogue['services']).issubset(domains):
                data.extend(parse_dialogue_into_examples(dialogue, context_len=context_len))

        save_data(data, f_name)

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


class DataModule:
    def __init__(self,
                 tokenizer_name: str,
                 label_column: str,
                 use_columns: List[str],
                 context_len: int = 1,
                 max_seq_length: int = None,
                 val_size: float = 0.3,
                 additional_special_tokens: List[str] = None,
                 cache_dir: str = "../huggingface_dataset",
                 domains: List[str] = None
                 ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.label_column = label_column
        self.use_columns = use_columns
        self.max_seq_length = max_seq_length
        self.context_len = context_len
        self.val_size = val_size
        self.domains = domains

        # Initialize pretrained tokenizer and register all the special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True,
                                                       additional_special_tokens=additional_special_tokens)

        print(f"Special tokens: {self.tokenizer.additional_special_tokens}")
        print(f"Domains: {self.domains}")

        # Load train/val/test datasets into DataFrames
        train_df = load_multiwoz_dataset('train', context_len=self.context_len, cache_dir=cache_dir, domains=domains)
        val_df = load_multiwoz_dataset('validation', context_len=self.context_len, cache_dir=cache_dir, domains=domains)
        test_df = load_multiwoz_dataset('test', context_len=self.context_len, cache_dir=cache_dir, domains=domains)

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


if __name__ == '__main__':
    print(f"CREATING DATASET...")
    """
    print(f"\t LOADING TRAIN DATA FROM \'{args.train_data}\'...\n"
          f"\t LOADING VAL DATA FROM \'{args.val_data}\'...\n"
          f"\t LOADING TEST DATA FROM \'{args.test_data}\'...\n"
          f"\t LOADING TOKENIZER FROM \'{args.pretrained_model}\'...\n")
    """
    special_tokens = [BELIEF, CONTEXT, USER]
    dm = DataModule(tokenizer_name=args.tokenizer_name,
                    label_column='actions',
                    use_columns=['actions', 'utterance'],
                    max_seq_length=args.max_seq_length,
                    additional_special_tokens=special_tokens,
                    cache_dir=args.cache_dir,
                    domains=args.domains)

    print("DATASET CREATED!")
    print(f"{50 * '='}")

    model_type = args.pretrained_model
    run_name = f"{model_type}_{datetime.now().strftime('%y%m%d-%H%M%S')}"
    output_dir = Path(args.save_folder) / model_type / run_name
    if args.train_model:
        print(f"TRAINING A NEW MODEL USING \'{args.pretrained_model}\'...")

        # Create TrainingArguments to access all the points of customization for the training.
        training_args = TrainingArguments(
            run_name=run_name,
            output_dir=output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            logging_dir=Path('logdir') / run_name,
            learning_rate=args.learning_rate,
            weight_decay=0,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            num_train_epochs=args.epochs,
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            save_total_limit=1,
            warmup_steps=300,
        )
        # Load pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model,
                                                                   num_labels=dm.num_labels,
                                                                   id2label=dm.id2label,
                                                                   label2id=dm.label2id,
                                                                   problem_type="multi_label_classification").to(device)

        model.resize_token_embeddings(len(dm.tokenizer))

        # Prepare model for training, i.e. this command does not train the model.
        model.train()

        # Create HuggingFace Trainer, which provides an API for feature-complete training in PyTorch for most
        # standard use cases. The API supports distributed training on multiple GPUs/TPUs, mixed precision through
        # NVIDIA Apex and Native AMP for PyTorch. The Trainer contains the basic training loop which supports the
        # above features.
        trainer = Trainer(model=model,
                          args=training_args,
                          train_dataset=dm.dataset['train'],
                          eval_dataset=dm.dataset['val'],
                          compute_metrics=compute_metrics,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])

        # Train our model using the trainer and save it after the training is complete.
        trainer.train()
        trainer.save_model()
        print("TRAINING A NEW MODEL DONE!")
        print(f"{50 * '='}")

    # =========================================================================================
    # Load model and create predictions
    if args.train_model:
        model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
        report_dir = output_dir
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model).to(device)
        report_dir = Path(args.pretrained_model)

    print(f"EVALUATING MODEL {report_dir}:")

    model.eval()

    # TODO: top_k set to what value? len(dm.label2id) is more than 200 and we don't want that, too long processing
    classifier_pipeline = pipelines.pipeline(task='text-classification', model=model, tokenizer=dm.tokenizer,
                                             top_k=5,
                                             device=0 if torch.cuda.is_available() else -1)

    for dataset_name, dataset_data in dm.dataset.items():
        print(f"EVALUATING MODEL ON {dataset_name}...")
        output_df = dataset_data.to_pandas()
        eval_start_time = time.time()
        predictions = classifier_pipeline(KeyDataset(dataset_data, 'text'))

        print(f"Predictions using model pipeline computed in {time.time() - eval_start_time} -> "
              f"Compare with true labels...")
        comparing_start_time = time.time()

        predicted_labels = []
        y_pred = np.zeros((len(predictions), len(dm.label2id)), dtype='bool')
        y_true = np.array(dataset_data['label'])

        predictions_times = []
        for i, prediction in tqdm(enumerate(predictions), total=len(predictions)):
            # i-th `prediction` is a model output for the i-th input dialogue.
            # It is a list of dict items with the following format:
            #   - len: number of predicted actions - top_k in the pipeline
            #   - elements: dict with the following key-value pairs:
            #       - prediction['label']: action name
            #       - prediction['score']: probability of this action
            prediction_start_time = time.time()
            labels = []
            labels_ids = []
            for pred in prediction:
                score = round(pred['score'], 4)
                action = pred['label']
                if score >= 0.5:
                    labels.append(action)
                    labels_ids.append(dm.label2id[action])

            predicted_labels.append(labels)
            y_pred[i, labels_ids] = 1
            predictions_times.append(time.time() - prediction_start_time)

        print(f"Compare with true labels done in: {time.time() - comparing_start_time}\n"
              f"Average time for one prediction is: {np.mean(predictions_times)}.")

        y_pred = y_pred.astype('float')
        output_df['predicted'] = predicted_labels
        output_df['scores'] = predictions
        output_df = output_df[['text', 'actions', 'predicted', 'system_utterance', 'scores']]

        os.makedirs(report_dir, exist_ok=True)
        output_df.to_csv(report_dir / f'{dataset_name}_predictions.csv')

        # =========================================================================================
        # Evaluate model

        actions_recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)
        weighted_recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
        macro_recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
        recall = {'metric': 'recall', 'macro': round(macro_recall, 4), 'weighted': round(weighted_recall, 4)} | \
                 {action: round(actions_recall[i], 4) for i, action in dm.id2label.items()}

        actions_precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
        weighted_precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
        macro_precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
        precision = {'metric': 'precision', 'macro': round(macro_precision, 4),
                     'weighted': round(weighted_precision, 4)} | \
                    {action: round(actions_precision[i], 4) for i, action in dm.id2label.items()}

        actions_f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
        weighted_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        f1 = {'metric': 'f1', 'macro': round(macro_f1, 4), 'weighted': round(weighted_f1, 4)} | \
             {action: round(actions_f1[i], 4) for i, action in dm.id2label.items()}

        accuracy = {'metric': 'accuracy', 'macro': round(accuracy_score(y_true=y_true, y_pred=y_pred), 4)}

        metrics_df = pd.DataFrame([accuracy, recall, precision, f1])
        print(metrics_df)
        metrics_df.to_csv(report_dir / f'{dataset_name}_metrics.csv')

        print(f"EVALUATING MODEL ON {dataset_name} DONE!\n"
              f"\t Total time: {time.time() - eval_start_time}")

    print("=======================================================")
    print(f"EVALUATING MODEL DONE!")
