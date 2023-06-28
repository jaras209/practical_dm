from typing import Dict, NewType

# Define your BeliefState type alias
BeliefState = NewType('BeliefState', Dict[str, Dict[str, str]])

