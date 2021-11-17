import json
import pandas as pd
import os
import numpy as np
import re


DOMAIN_NAMES = ['Restaurant', 'Attraction', 'Hotel', 'Taxi', 'Train', 'Bus', 'Hospital', 'Police']


def inspect_data(dialogues: str, acts: str):
    with open(dialogues, 'r') as file:
        dialogues = json.load(file)

    with open(acts, 'r') as file:
        acts = json.load(file)

    # 1. list index: dialogue
    dialogue = dialogues[0]
    # 2. dict index: dialogue information,
    #   'services' gets a list of strings with domains
    #   'turns' gets a list of dialogue turns
    #   'dialogue_id' gets dialogue ID for actions
    turns = dialogue['turns']
    dialogue_id = dialogue['dialogue_id']

    """
    for i in range(len(turns)):
        # 3. list index: turn id
        turn = turns[i]
        # 4 dict index: part of the turn,
        #   - 'utterance' gets the utterance itself
        #   - 'turn_id' gets id of the turn
        #   - 'speaker' is the "USER" or the "SYSTEM"
        utterance = turn['utterance']
        turn_id = turn['turn_id']
        speaker = turn['speaker']

        # acts is dictionary with keys "$dialogue_id".
        #   `acts["$dialogue_id"]` is again a dictionary with keys being string of "$turn_id".
        #       `acts["$dialogue_id"]["$turn_id"]` is again a dictionary with 2 keys: 'dialog_act', 'span_info'.
        #           - `acts["$dialogue_id"]["$turn_id"]['dialog_act']` is again a dict with "$act_name" as keys.
        #               -- each such key has a list of lists with 2 values:
        #                       >> ["$slot_name", "$action_value"]
        #           - `acts["$dialogue_id"]["$turn_id"]['span_info']` is a list of lists with 5 values:
        #               -- ["$act_name", "$slot_name", "$action_value",
        #                   "$start_charater_index", "$exclusive_end_character_index"]
        #
        '''
        {
          "$dialogue_id": [
            "$turn_id": {
              "dialogue_acts": {
                "$act_name": [
                  [
                    "$slot_name",
                    "$action_value"
                  ]
                ]
              },
              "span_info": [
                [
                  "$act_name"
                  "$slot_name",
                  "$action_value"
                  "$start_charater_index",
                  "$exclusive_end_character_index"
                ]
              ]
            }
          ]
        }
        '''

        action = acts[dialogue_id][f'{i}']['dialog_act']
        print(f'ID: {turn_id} - SPEAKER: {speaker} - UTTERANCE: {utterance} - ACTION: {action}')
    """

    # Let us go through all dialogues and all turns and gather all slot names
    system_slots = set()
    with open('../data/dialogue_info.csv', 'w') as file:
        for dialogue in dialogues:
            turns = dialogue['turns']
            dialogue_id = dialogue['dialogue_id']
            for i, turn in enumerate(turns):
                utterance = turn['utterance']
                turn_id = turn['turn_id']
                speaker = turn['speaker']
                action = acts[dialogue_id][f'{i}']['dialog_act']

                if speaker == 'SYSTEM':
                    for act_name, values in action.items():
                        for pair in values:
                            slot, _ = pair
                            system_slots.add(f'{act_name}-{slot}')

                print(f'TURN_ID: {turn_id} - SPEAKER: {speaker} - UTTERANCE: {utterance} - ACTION: {action}', file=file)

            print('================================== NEXT DIALOGUE ======================================', file=file)

    with open('../data/dialogue_actions.txt', 'w') as file:
        print(f'There are {len(system_slots)} unique actions:', file=file)
        for slot in system_slots:
            print(slot, file=file)


def extract_data(dialogue_dir: str, acts: str, strip_domain: bool = False):
    dialogue_files = os.listdir(dialogue_dir)
    data = []
    with open(acts, 'r') as file:
        acts = json.load(file)

    for dialogue_file in dialogue_files:
        with open(dialogue_dir + '/' + dialogue_file, 'r') as file:
            dialogues = json.load(file)

        row = None
        for dialogue in dialogues:
            turns = dialogue['turns']
            dialogue_id = dialogue['dialogue_id']
            history = []
            for i, turn in enumerate(turns):
                utterance = re.sub(r'[^\w\s]', '', turn['utterance']).split()
                turn_id = turn['turn_id']
                speaker = turn['speaker']
                actions = acts[dialogue_id][f'{i}']['dialog_act']

                if speaker == 'USER':
                    row = {'user_utterance': utterance}

                else:
                    act_name_slot_pairs = []
                    for act_name, values in actions.items():
                        if strip_domain:
                            act_name = '-'.join([x for x in act_name.split('-') if x not in DOMAIN_NAMES])
                        for slot_value_pair in values:
                            slot, _ = slot_value_pair
                            act_name_slot_pairs.append(f'{act_name}{"-" if slot != "none" else ""}'
                                                       f'{slot if slot != "none" else ""}')

                    row.update({
                        'system_utterance': utterance,
                        'system_actions': act_name_slot_pairs,
                        'history': history.copy(),
                    })
                    data.append(row)
                    history += row['user_utterance'] + row['system_utterance']
    return data


class Dataset:
    PAD, UNK = 0, 1

    # Functions for mapping words and system_actions names into integers
    def words2int(self, words):
        output = []
        for word in words:
            output.append(self.word2int_vocabulary.get(word, Dataset.UNK))

        return output

    def actions2int(self, actions):
        output = []
        for action in actions:
            output.append(self.actions2int_vocabulary.get(action, 0))

        return output

    def ints2words(self, ints):
        output = []
        for word in ints:
            output.append(self.int2word_vocabulary.get(word, '<UNK>'))

        return output

    def ints2actions(self, actions):
        output = []
        actions_indices = np.nonzero(actions)[0]
        for action in actions_indices:
            output.append(self.int2action_vocabulary.get(action, '<UNK_ACT>'))

        return output

    def create_dataset(self, data, num_actions: int):
        size = len(data)
        max_utt_len = max(len(row['user_utterance']) for row in data)
        max_hist_len = max(len(row['history']) for row in data)
        utterances = np.zeros((size, max_utt_len), np.int32)
        actions = np.zeros((size, num_actions), np.int32)
        history = np.zeros((size, max_hist_len), np.int32)
        for i, row in enumerate(data):
            u = self.words2int(row['user_utterance'])
            a = self.actions2int(row['system_actions'])
            h = self.words2int(row['history'])
            actions[i, a] = 1
            utterances[i, :len(u)] = u
            history[i, :len(h)] = h

        return utterances, actions, history

    def __init__(self, train_folder, val_folder, test_folder, actions_file, strip_domain=True, save_folder=None):
        # Load the data
        train_data = extract_data(train_folder, actions_file, strip_domain=strip_domain)
        val_data = extract_data(val_folder, actions_file, strip_domain=strip_domain)
        test_data = extract_data(test_folder, actions_file, strip_domain=strip_domain)

        # TODO: určite je zapotřebí vyhledávat v utterancích entity a ty mapovat na nějaké placeholdery, např. čas,
        #  jména atd.

        self.word2int_vocabulary = {'<PAD>': Dataset.PAD, '<UNK>': Dataset.UNK}
        i = 2
        for row in train_data:
            for word in row['user_utterance'] + row['system_utterance']:
                if word not in self.word2int_vocabulary.keys():
                    self.word2int_vocabulary[word] = i
                    i += 1

        self.num_words = len(self.word2int_vocabulary)
        self.int2word_vocabulary = {v: k for k, v in self.word2int_vocabulary.items()}

        # Create map from system_actions names into integers
        unique_actions = sorted(list(set([action for row in train_data for action in row['system_actions']])))
        self.num_actions = len(unique_actions)
        self.actions2int_vocabulary = {v: k for k, v in enumerate(unique_actions)}
        self.int2action_vocabulary = {v: k for k, v in self.actions2int_vocabulary.items()}

        # Save dictionaries into a folder for later reuse.
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            with open(f'{save_folder}/word2int_vocabulary.json', 'w') as f:
                json.dump(self.word2int_vocabulary, f, indent=2)

            with open(f'{save_folder}/actions2int_vocabulary.json', 'w') as f:
                json.dump(self.actions2int_vocabulary, f, indent=2)

        # Create train, val and test data
        self.train_utterances, self.train_actions, self.train_history = self.create_dataset(train_data, self.num_actions)
        self.val_utterances, self.val_actions, self.val_history = self.create_dataset(val_data, self.num_actions)
        self.test_utterances, self.test_actions, self.test_history = self.create_dataset(test_data, self.num_actions)

