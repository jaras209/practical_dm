import json
import random
import copy
import os
from pathlib import Path
from typing import Text, Dict, Optional, Union, List
from fuzzywuzzy import fuzz


class MultiWOZDatabase:
    """ MultiWOZ database implementation. """

    IGNORE_VALUES = {
        'hospital': ['id'],
        'police': ['id'],
        'attraction': ['location', 'openhours'],
        'hotel': ['location', 'price'],
        'restaurant': ['location', 'introduction']
    }

    FUZZY_KEYS = {
        'hospital': {'department'},
        'hotel': {'name'},
        'attraction': {'name'},
        'restaurant': {'name', 'food'},
        'bus': {'departure', 'destination'},
        'train': {'departure', 'destination'},
        'police': {'name'}
    }

    DOMAINS = [
        'restaurant',
        'hotel',
        'attraction',
        'train',
        'taxi',
        'police',
        'hospital'
    ]

    def __init__(self, database_path: Path):
        self.data, self.data_keys = self._load_data(database_path)

    def _load_data(self, database_path: Path):
        database_data = {}
        database_keys = {}

        for domain in self.DOMAINS:
            with open(database_path / f"{domain}_db.json", "r") as db_file:
                for line in db_file:
                    if not line.startswith('##') and line.strip() != "":
                        db_file.seek(0)
                        break
                database_data[domain] = json.load(db_file)

            if domain in self.IGNORE_VALUES:
                for database_item in database_data[domain]:
                    for ignore in self.IGNORE_VALUES[domain]:
                        if ignore in database_item:
                            database_item.pop(ignore)

            database_keys[domain] = set()
            if domain == 'taxi':
                database_data[domain] = {k.lower(): v for k, v in database_data[domain].items()}
                database_keys[domain].update([k.lower() for k in database_data[domain].keys()])
            else:
                for i, database_item in enumerate(database_data[domain]):
                    database_data[domain][i] = {k.lower(): v for k, v in database_item.items()}
                    database_keys[domain].update([k.lower() for k in database_item.keys()])

        return database_data, database_keys

    def time_str_to_minutes(self, time_string) -> Text:
        """ Convert the time string into a number (number of minutes, hours, seconds from 1970, whatever you want) """
        time = time_string
        # Copied from https://github.com/Tomiinek/MultiWOZ_Evaluation/blob/4e60b60e58ff885412b630af3a86ad1f63135118/mwzeval/normalization.py#L165-L219
        """ Converts time to the only format supported by database, e.g. 07:15. """
        time = time.strip().lower()

        if time == "afternoon":
            return "13:00"
        if time == "lunch" or time == "noon" or time == "mid-day" or time == "around lunch time":
            return "12:00"
        if time == "morning":
            return "08:00"
        if time.startswith("one o'clock p.m"):
            return "13:00"
        if time.startswith("ten o'clock a.m"):
            return "10:00"
        if time == "seven o'clock tomorrow evening":
            return "07:00"
        if time == "three forty five p.m":
            return "15:45"
        if time == "one thirty p.m.":
            return "13:30"
        if time == "six fourty five":
            return "06:45"
        if time == "eight thirty":
            return "08:30"

        if time.startswith("by"):
            time = time[3:]

        if time.startswith("after"):
            time = time[5:].strip()

        if time.startswith("afer"):
            time = time[4:].strip()

        if time.endswith("am"):
            time = time[:-2].strip()
        if time.endswith("a.m."):
            time = time[:-4].strip()

        if time.endswith("pm") or time.endswith("p.m."):
            if time.endswith("pm"):
                time = time[:-2].strip()
            if time.endswith("p.m."):
                time = time[:-4].strip()
            tokens = time.split(':')
            if len(tokens) == 2:
                return str(int(tokens[0]) + 12) + ':' + tokens[1]
            if len(tokens) == 1 and tokens[0].isdigit():
                return str(int(tokens[0]) + 12) + ':00'

        if len(time) == 0:
            return "00:00"

        if time[-1] == '.' or time[-1] == ',' or time[-1] == '?':
            time = time[:-1]

        if time.isdigit() and len(time) == 4:
            return time[:2] + ':' + time[2:]

        if time.isdigit(): return time.zfill(2) + ":00"

        if ':' in time:
            time = ''.join(time.split(' '))

        if len(time) == 4 and time[1] == ':':
            tokens = time.split(':')
            return tokens[0].zfill(2) + ':' + tokens[1]

        return time

    def query(self, domain: str, constraints: Dict[str, str], fuzzy_ratio: int = 90) -> Optional[List[Dict[str, str]]]:
        """
        Query the database based on the specified constraints for a given domain.

        This function filters and returns the list of entities (dictionaries) that match the constraints for the specified domain.
        It also handles the constraints with 'None' values or non-existing keys by ignoring them.
        It also provides an optional parameter to specify a fuzzy matching ratio for the matching process.

        Arguments:
            domain (str): Domain of the data ('hotel', 'restaurant', 'attraction', 'train', or 'hospital')
            constraints (Dict[str, str]): Hard constraints to the query results. The constraints should be a dictionary where
                                           keys are the domain-specific properties and values are the expected values for these properties.
            fuzzy_ratio (int, optional): A threshold to control the fuzziness of the match. It is useful when matching string properties.
                                         The value should be between 0 (lowest match) and 100 (exact match). Defaults to 90.

        Returns:
            list[dict] or None: List of dictionaries with data entries that meet the constraints, None if the domain
            was not queried at all.
        """

        # Remove domain from the constraint keys if present and exclude keys mapped to "None"
        clean_constraints = {}
        for k, v in constraints.items():
            if v != "None":  # Exclude keys mapped to "None"
                clean_constraints[k.replace(domain + '-', '')] = v

        constraints = clean_constraints

        # Check if all values in cleaned constraints are "None". If they are, return None immediately,
        # indicating the domain was not queried at all.
        if not clean_constraints:
            return None

        if domain == 'taxi':
            taxi_color, taxi_type, taxi_phone = None, None, None

            # Fetch taxi color from constraints or choose a random color if not specified
            taxi_color = constraints.get('color')
            if taxi_color is None:
                taxi_color = random.choice(self.data[domain]['taxi_colors'])

            # Fetch a taxi type from constraints or choose a random type if not specified
            taxi_type = constraints.get('type')
            if taxi_type is None:
                taxi_type = random.choice(self.data[domain]['taxi_types'])

            # Fetch taxi phone from constraints or generate a random phone number if not specified
            taxi_phone = constraints.get('phone')
            if taxi_phone is None:
                taxi_phone = ''.join([str(random.randint(1, 9)) for _ in range(11)])

            return [{'color': taxi_color, 'type': taxi_type, 'phone': taxi_phone}]

        elif domain == 'hospital':

            # Define basic hospital details
            default_hospital = {
                'hospital phone': '01223245151',
                'address': 'Hills Rd, Cambridge',
                'postcode': 'CB20QQ',
                'name': 'Addenbrookes'
            }

            # If 'department' constraint is present, ensure it is a list
            constraint_department = constraints.get('department')
            if constraint_department is not None:
                department_list = [constraint_department] if isinstance(constraint_department,
                                                                        str) else constraint_department
                department_list = [department.strip().lower() for department in department_list]
            else:
                department_list = []

            # If 'phone' constraint is present, ensure it is a list
            constraint_phone = constraints.get('phone')
            if constraint_phone is not None:
                phone_list = [constraint_phone.strip().lower()]
            else:
                phone_list = []

            # If no department or phone constraints are present, return the default hospital
            if len(department_list) == 0 and len(phone_list) == 0:
                return [dict(default_hospital)]
            else:
                query_results = []
                for data_item in self.data[domain]:  # Iterate over all data items in the 'hospital' domain
                    if 'department' in self.FUZZY_KEYS[domain]:  # Check if 'department' should be fuzzy matched
                        match_func = (lambda x: fuzz.partial_ratio(data_item['department'].lower(), x) > fuzzy_ratio)
                    else:
                        match_func = (lambda x: data_item['department'].lower() == x)

                    # If any department matches and phone constraints are met, append the result
                    if any(match_func(department) for department in department_list) and \
                            (len(phone_list) == 0 or any(data_item['phone'] == phone for phone in phone_list)):
                        result_item = dict(data_item)
                        result_item.update(default_hospital)  # Add the default hospital details to the result
                        query_results.append(result_item)

                return query_results

        elif domain in self.DOMAINS:
            # Hotel database keys:
            #   address, area, name, phone, postcode, pricerange, type, internet, parking, stars, takesbookings
            #       (other are ignored)

            # Attraction database keys:
            #   address, area, name, phone, postcode, pricerange, type, entrance fee (other are ignored)

            # Restaurant database keys:
            #   address, area, name, phone, postcode, pricerange, type, food

            # Train database contains keys:
            #   arriveby, departure, day, leaveat, destination, trainid, price, duration
            #       The keys arriveby, leaveat expect a time format such as 8:45 for 8:45 am

            # If the 'entrancefee' constraint is present in 'attraction' domain, rename it to 'entrance fee'
            if domain == 'attraction' and 'entrancefee' in constraints:
                constraints['entrance fee'] = constraints.pop('entrancefee')

            # Initialize empty query dictionary
            query_dict = {}

            # For each key in the domain, if the key exists in the constraints and is not empty, add it to the query
            # dictionary
            for key in self.data_keys[domain]:
                constraint_value = constraints.get(key)
                if constraint_value is not None and constraint_value != 'None':
                    if isinstance(constraint_value, str):
                        constraint_value = [constraint_value]
                    if key in ['arriveby', 'leaveat']:
                        constraint_value = [self.time_str_to_minutes(time) for time in constraint_value]
                        constraint_value = list(set(constraint_value))
                    query_dict[key] = constraint_value

            # Initialize result list
            query_results = []

            # Iterate over every item in the domain's data
            for i, data_item in enumerate(self.data[domain]):
                # Iterate over every key, value pair in the query
                for key, values in query_dict.items():
                    # If the key is not in the data item, or its value is '?' or empty, continue to the next iteration
                    if key not in data_item or data_item[key] == '?' or values is None or len(values) == 0:
                        continue

                    # For 'arriveby' key, the data item value should be earlier than the query times
                    if key == 'arriveby':
                        if not any(data_item[key] != ":" and data_item[key] < time for time in values):
                            break

                    # For 'leaveat' key, the data item value should be later than the query times
                    elif key == 'leaveat':
                        if not any(data_item[key] != ":" and data_item[key] > time for time in values):
                            break

                    # For other keys, the data item value should match the query values
                    else:
                        # If fuzzy matching is to be used for the key, define a match function using fuzz.partial_ratio
                        # Otherwise, define a match function for exact equality
                        if key in self.FUZZY_KEYS[domain]:
                            match_func = (lambda x: fuzz.partial_ratio(data_item[key].lower(), x) > fuzzy_ratio)
                        else:
                            match_func = (lambda x: data_item[key].lower() == x)

                        # If none of the query values match the data item value, break the loop
                        if not any(match_func(value.strip().lower()) for value in values):
                            break

                # If the loop finished without breaking, all constraints were satisfied
                else:
                    result = copy.deepcopy(data_item)
                    if domain in ['train', 'hotel', 'restaurant']:
                        ref = constraints.get('ref')
                        if ref is not None and ref != 'None':
                            result['ref'] = '{0:08d}'.format(i) if ref == "" else ref
                    query_results.append(result)

            # If the domain is 'attraction', rename 'entrance fee' back to 'entrancefee'
            if domain == 'attraction':
                for result in query_results:
                    result['entrancefee'] = result.pop('entrance fee')

            return query_results
