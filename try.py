import json

if __name__ == "__main__":
    with open('dialogues_001.json', 'r') as file:
        data = json.load(file)

    with open('dialog_acts.json', 'r') as file:
        acts = json.load(file)

    # 1. list index: dialogue
    dialogue = data[0]
    # 2. dict index: dialogue information,
    #   'turns' gets a list of dialogue turns
    #   'dialogue_id' gets dialogue ID for actions
    turns = dialogue['turns']
    dialogue_id = dialogue['dialogue_id']
    
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

    # Let us go through all dialogues and all turns and gather all slot names

    system_slots = set()
    with open('dialogue_info.txt', 'w') as file:
        for dialogue in data:
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

            print('====================================== NEXT DIALOGUE =======================================', file=file)

    with open('dialogue_actions.txt', 'w') as file:
        print(f'There are {len(system_slots)} unique actions:', file=file)
        for slot in system_slots:
            print(slot, file=file)


