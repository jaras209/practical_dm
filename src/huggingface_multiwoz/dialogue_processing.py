import copy
import logging
from typing import Dict, Any, Optional, List, Tuple

from huggingface_multiwoz.constants import DOMAIN_NAMES, DOMAIN_SLOTS
from huggingface_multiwoz.database import MultiWOZDatabase

from typing import Dict, List


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


def get_database_results(database: MultiWOZDatabase,
                         belief_state: Dict[str, Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get the database results for the current domain and belief state.

    Args:
        database (MultiWOZDatabase): The database instance used for querying.
        belief_state (Dict[str, Dict[str, str]]): The current belief state.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary with domain as the key and a list of resulting entities as the value.
    """
    database_results = {}

    for domain, domain_state in belief_state.items():
        results = database.query(domain, domain_state)
        database_results[domain] = results

    return database_results


def process_user_turn(turn_id: int,
                      turns: Dict[str, List],
                      belief_state: Dict[str, Dict[str, str]],
                      dialogue_domain: str,
                      database: MultiWOZDatabase) -> Tuple[Dict, Dict[str, Dict[str, str]]]:
    """
    Processes a user turn by updating the belief state, state update, and database results based on the user's utterance.

    Args:
        turn_id (int): The index of the current turn in the dialogue.
        turns (Dict[str, List]): A dictionary containing the turns data from the dialogue.
        belief_state (Dict[str, Dict[str, str]]): The current belief state.
        dialogue_domain (str): The domain of the dialogue.
        database (MultiWOZDatabase): The MultiWOZ database object.

    Returns:
        Tuple[Dict, Dict[str, Dict[str, str]]]: A tuple containing the updated example dictionary and the
                                                updated belief state.
    """
    # logging.debug(f"Processing system turn with turn_id={turn_id}")

    # Extract the user's utterance
    utterance = turns['utterance'][turn_id]

    # Create an example with the user's utterance and the current belief state
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
    # database_results = get_database_results(database, belief_state)

    # Update the example with the new belief state and database results
    example.update({
        'new_belief_state': copy.deepcopy(belief_state),
        # 'state_update': state_update,
        # 'database_results': database_results
    })

    return example, belief_state


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


def process_system_turn(turn_id: int,
                        turns: Dict[str, Any],
                        example: Dict[str, Any],
                        context_len: Optional[int] = None,
                        strip_domain: bool = False) -> Dict[str, Any]:
    """
    Process a system turn in the dialogue, updating the example dictionary with the relevant information.

    Args:
        turn_id (int): The index of the current turn in the dialogue.
        turns (dict): A dictionary containing the turns data from the dialogue.
        example (dict): The example dictionary to be updated with the system turn information.
        context_len (int, optional): The maximum number of utterances to include in the context. If None, all
                                      utterances up to the current turn are included. Defaults to None.
        strip_domain (bool, optional): If True, remove the domain names from the act types. Defaults to False.

    Returns:
        dict: The updated example dictionary with the system turn information.
    """
    logging.debug(f"Processing system turn with turn_id={turn_id}")

    # Get the current system utterance and dialogue acts
    utterance = turns['utterance'][turn_id]
    dialogue_acts = turns['dialogue_acts'][turn_id]

    # Extract action types and slot names from the dialogue acts
    act_type_slot_name_pairs = extract_act_type_slot_name_pairs(dialogue_acts, strip_domain=strip_domain)

    # Create the dialogue context
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

    return example


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

        # USER
        if speaker == 0:
            # Process user turn and update belief state
            example, belief_state = process_user_turn(turn_id, turns, belief_state, dialogue_domain, database)
            print(belief_state)
        # SYSTEM
        else:
            # Process system turn and create an example
            example = process_system_turn(turn_id, turns, example, context_len=context_len, strip_domain=strip_domain)
            # Append the example to the list of examples
            examples.append(example)

    return examples
