UNK_ACTION = 'UNK'
BELIEF = '<|BELIEF|>'
CONTEXT = '<|CONTEXT|>'
USER = '<|USER|>'
DATABASE = '<|DATABASE|>'
DATABASE_COUNTS = '<|DATABASE_COUNTS|>'
SEPARATOR_BELIEF_UPDATE = 'Next utterance:'
TASK_DESCRIPTION_BELIEF_UPDATE = 'Update belief state'
SPECIAL_TOKENS = [BELIEF, DATABASE, DATABASE_COUNTS, CONTEXT, USER]
DOMAIN_NAMES = sorted({'hotel', 'restaurant', 'bus', 'train', 'attraction', 'hospital', 'taxi'})
OUTPUT_DF_COLUMNS = ['text', 'actions', 'predicted', 'dialogue_acts', 'system_utterance', 'scores']
DOMAIN_SLOTS = dict(bus=sorted(["day", "departure", "destination", "leaveat"]),
                    hotel=sorted(["area", "bookday", "bookpeople", "bookstay", "internet", "name", "parking",
                                  "pricerange", "stars", "type"]),
                    restaurant=sorted(["area", "bookday", "bookpeople", "booktime", "food", "name", "pricerange"]),
                    train=sorted(["arriveby", "bookpeople", "day", "departure", "destination", "leaveat"]),
                    attraction=sorted(["area", "name", "type"]),
                    hospital=["department"],
                    taxi=sorted(["arriveby", "departure", "destination", "leaveat"]))
