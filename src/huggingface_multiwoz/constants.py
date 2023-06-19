UNK_ACTION = 'UNK'
BELIEF = '<|BELIEF|>'
CONTEXT = '<|CONTEXT|>'
USER = '<|USER|>'
DATABASE = '<|DATABASE|>'
DATABASE_COUNTS = '<|DATABASE_COUNTS|>'
SEP_TOKEN_BELIEF = 'Next utterance:'
TASK_DESCRIPTION = 'Update belief state'
SPECIAL_TOKENS = [BELIEF, DATABASE, DATABASE_COUNTS, CONTEXT, USER]
DOMAIN_NAMES = sorted({'hotel', 'restaurant', 'police', 'bus', 'train', 'attraction', 'hospital', 'taxi'})
OUTPUT_DF_COLUMNS = ['text', 'actions', 'predicted', 'dialogue_acts', 'system_utterance', 'scores']
# TODO: probably need to verify that these are all the slots
DOMAIN_SLOTS = dict(bus=sorted({'bus-departure', 'bus-destination', 'bus-leaveat', 'bus-day'}),
                    hotel=sorted({'hotel-pricerange', 'hotel-bookstay', 'hotel-bookday', 'hotel-area', 'hotel-stars',
                                  'hotel-bookpeople', 'hotel-parking', 'hotel-type', 'hotel-name', 'hotel-internet'}),
                    restaurant=sorted({'restaurant-bookpeople', 'restaurant-booktime', 'restaurant-pricerange',
                                       'restaurant-name', 'restaurant-area', 'restaurant-bookday', 'restaurant-food',
                                       }),
                    police={'something_police'},
                    train=sorted({'train-departure', 'train-bookpeople', 'train-destination', 'train-leaveat',
                                  'train-day', 'train-arriveby'}),
                    attraction=sorted({'attraction-type', 'attraction-area', 'attraction-name'}),
                    hospital={'hospital-department'},
                    taxi=sorted({'taxi-departure', 'taxi-destination', 'taxi-arriveby', 'taxi-leaveat'}))
