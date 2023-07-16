UNK_ACTION = 'UNK'
BELIEF = '[state]'
CONTEXT = '[context]'
USER = '[user]'
DATABASE = '[database]'
DATABASE_COUNTS = '[database]'
CONTEXT_SEP = '. '
STATE_GEN = '[state]'
CONTEXT_GEN = '[context]'
USER_GEN = '[user]'
DATABASE_COUNTS_GEN = '[database]'
TASK_DESCRIPTION_STATE_UPDATE = 'Update state:'
TASK_DESCRIPTION_ACTION_GENERATION = 'Generate actions:'
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
