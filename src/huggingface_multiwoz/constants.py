UNK_ACTION = 'UNK'
BELIEF = '<|BELIEF|>'
CONTEXT = '<|CONTEXT|>'
USER = '<|USER|>'
DATABASE = '<|DATABASE|>'
DATABASE_COUNTS = '<|DATABASE_COUNTS|>'
CONTEXT_SEP = '. '
BELIEF_GEN = 'Belief:'
CONTEXT_GEN = 'Context:'
USER_GEN = 'User:'
DATABASE_COUNTS_GEN = 'Database:'
TASK_DESCRIPTION_BELIEF_UPDATE = 'Update Belief:'
TASK_DESCRIPTION_ACTION_GENERATION = 'Select actions:'
SPECIAL_TOKENS = [BELIEF, DATABASE, DATABASE_COUNTS, CONTEXT, USER]
DOMAIN_NAMES = sorted({'hotel', 'restaurant', 'bus', 'train', 'attraction', 'hospital', 'taxi'})
OUTPUT_DF_COLUMNS = ['text', 'actions', 'predicted', 'dialogue_acts', 'system_utterance', 'scores']
COMMON_SLOTS = ["intent", "requested"]
DOMAIN_SLOTS = dict(bus=COMMON_SLOTS + sorted(["day", "departure", "destination", "leaveat"]),
                    hotel=COMMON_SLOTS + sorted(
                        ["area", "bookday", "bookpeople", "bookstay", "internet", "name", "parking",
                         "pricerange", "stars", "type"]),
                    restaurant=COMMON_SLOTS + sorted(
                        ["area", "bookday", "bookpeople", "booktime", "food", "name", "pricerange"]),
                    train=COMMON_SLOTS + sorted(
                        ["arriveby", "bookpeople", "day", "departure", "destination", "leaveat"]),
                    attraction=COMMON_SLOTS + sorted(["area", "name", "type"]),
                    hospital=COMMON_SLOTS + ["department"],
                    taxi=COMMON_SLOTS + sorted(["arriveby", "departure", "destination", "leaveat"]))
