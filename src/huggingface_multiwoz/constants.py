UNK_ACTION = 'UNK'
BELIEF = '<|BELIEF|>'
CONTEXT = '<|CONTEXT|>'
USER = '<|USER|>'
DATABASE = '<|DATABASE|>'
DATABASE_COUNTS = '<|DATABASE_COUNTS|>'
SPECIAL_TOKENS = [BELIEF, DATABASE, DATABASE_COUNTS, CONTEXT, USER]
DOMAIN_NAMES = {'hotel', 'restaurant', 'police', 'bus', 'train', 'attraction', 'hospital', 'taxi'}
# TODO: probably need to verify that these are all the slots
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