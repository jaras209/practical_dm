UNK_ACTION = 'UNK'
BELIEF = '<|BELIEF|>'
CONTEXT = '<|CONTEXT|>'
USER = '<|USER|>'
DATABASE = '<|DATABASE|>'
DATABASE_COUNTS = '<|DATABASE_COUNTS|>'
SEPARATOR_BELIEF_UPDATE = 'Next utterance:'
TASK_DESCRIPTION_BELIEF_UPDATE = 'Update belief state'
SPECIAL_TOKENS = [BELIEF, DATABASE, DATABASE_COUNTS, CONTEXT, USER]
DOMAIN_NAMES = sorted({'hotel', 'restaurant', 'police', 'bus', 'train', 'attraction', 'hospital', 'taxi'})
OUTPUT_DF_COLUMNS = ['text', 'actions', 'predicted', 'dialogue_acts', 'system_utterance', 'scores']
# TODO: probably need to verify that these are all the slots
DOMAIN_SLOTS = dict(bus=sorted({'departure', 'destination', 'leaveat', 'day'}),
                    hotel=sorted({'pricerange', 'bookstay', 'bookday', 'area', 'stars','bookpeople', 'parking', 'type',
                                  'name', 'internet'}),
                    restaurant=sorted({'bookpeople', 'booktime', 'pricerange', 'name', 'area', 'bookday', 'food'}),
                    train=sorted({'departure', 'bookpeople', 'destination', 'leaveat', 'day', 'arriveby'}),
                    attraction=sorted({'type', 'area', 'name'}),
                    hospital={'department'},
                    taxi=sorted({'departure', 'destination', 'arriveby', 'leaveat'}))
