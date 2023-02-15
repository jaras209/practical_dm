import pickle
import os
from pathlib import Path
import datasets
import torch.utils.data as torchdata
import transformers
import copy

DOMAIN_NAMES = {'hotel', 'restaurant', 'police', 'bus', 'train', 'attraction', 'hospital', 'taxi'}


class Dataset(torchdata.Dataset):
    """
    Dataset class, inherits from torch.utils.data.Dataset.
    Load the MultiWoz dataset using huggingface.datasets.
    Able to shorten the context length by setting context_len.
    """

    def __init__(self, split: str, context_len=None, cache_dir="../huggingface_dataset"):
        cache_dir = Path(cache_dir)
        print(Path(cache_dir).absolute())

        self.split = split
        self.fields = {}

        # Create cache dir if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.f_name = os.path.join(cache_dir, f"{split}_preprocessed_data.json")

        # self.database = MultiWOZDatabase()

        # If the dataset has already been preprocessed, load it from the cache
        if os.path.isfile(self.f_name):
            data = pickle.load(open(self.f_name, 'rb'))
            print(f"Loaded {len(data)} examples from cached file.")

        else:
            multi_woz_dataset = datasets.load_dataset(path='multi_woz_v22', split=split, ignore_verifications=True,
                                                      streaming=True)
            data = []
            for idx, dialogue in enumerate(multi_woz_dataset):
                if idx % 500 == 0:
                    print(f"Processing dialogue {idx + 1}")

                data.extend(self.parse_dialogue_into_examples(dialogue, context_len=context_len))

                if idx >= 1:
                    break
            # self.save_data(data)

        self.data = data

    def save_data(self, data):
        assert not os.path.exists(self.f_name), f"{self.f_name} already exists."
        with open(self.f_name, 'wb+') as f:
            pickle.dump(data, f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def parse_dialogue_into_examples(self, dialogue, context_len: int = None, strip_domain: bool = False):
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
        # print(turns)

        example = dict()
        belief_state = dict()
        for turn_id, _ in enumerate(turns['turn_id']):
            speaker = turns['speaker'][turn_id]
            dialogue_acts = turns['dialogue_acts'][turn_id]
            utterance = turns['utterance'][turn_id]

            # print(turn_id)

            """
            dialogue_acts.keys() = dict_keys(['dialog_act', 'span_info'])
                - values are again dictionaries:
                    dialogue_acts['dialog_act'].keys() = dict_keys(['act_type', 'act_slots'])
                    dialogue_acts['dialog_act'].values() = dict_values([[str], [{str: [str], str: [str}]])
                        dialogue_acts['dialog_act']['act_slots'].keys() = dict_keys(['slot_name', 'slot_value'])
                        dialogue_acts['dialog_act']['act_slots'].values() = dict_values([str], [str])
                        
                    dialogue_acts['span_info'].keys() = dict_keys(['act_type', 'act_slot_name', 'act_slot_value', 'span_start', 'span_end'])
                    dialogue_acts['span_info'].values() = dict_values([[str], [str], [str], [str], [str]])
                    
                        
                - example:   
                    dialogue_acts['dialog_act'] = {'act_type': ['Train-Inform'], 'act_slots': [{'slot_name': ['departure', 'destination'], 'slot_value': ['norwich', 'cambridge']}]}
                    dialogue_acts['span_info'] = {'act_type': ['Train-Inform', 'Train-Inform'], 'act_slot_name': ['departure', 'destination'], 'act_slot_value': ['norwich', 'cambridge'], 'span_start': [31, 42], 'span_end': [38, 51]}

            """

            # USER
            if speaker == 0:
                frame = turns['frames'][turn_id]
                """
                for state in frame['state']:
                    print(f"{state.keys() = }")
                    print(f"{state.values() = }")
                for slot in frame['slots']:
                    print(f"{slot.keys() = }")
                    print(f"{slot.values() = }")
                
                Frames:
                    frame.keys() = dict_keys(['service', 'state', 'slots'])
                    frame.values() = dict_values([['train'], [{'active_intent': 'find_train', 'requested_slots': [], 'slots_values': {'slots_values_name': ['train-departure', 'train-destination'], 'slots_values_list': [['norwich'], ['cambridge']]}}], [{'slot': [], 'value': [], 'start': [], 'exclusive_end': [], 'copy_from': [], 'copy_from_value': []}]])

                """
                domains = frame['service']
                states = frame['state']

                for domain, state in zip(domains, states):
                    slots = state['slots_values']['slots_values_name']
                    values = state['slots_values']['slots_values_list']

                    slot_value_pairs = {slot: value[0] for slot, value in zip(slots, values)}

                    if domain in belief_state.keys():
                        belief_state[domain].update(slot_value_pairs)
                    else:
                        belief_state[domain] = slot_value_pairs

                example = {
                    'utterance': utterance,
                    'belief_state':  copy.deepcopy(belief_state),
                }

            # SYSTEM
            else:
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
                    'actions': act_type_slot_name_pairs,
                    'system_utterance': utterance,
                })
                examples.append(example)
                print(f"{example = }")

        return examples


name = "test"

dataset = Dataset(name, context_len=None)
for i, item in enumerate(dataset):
    print(item)

# data_loader = DataLoader(dataset, batch_size=5, shuffle=True, collate=False)
