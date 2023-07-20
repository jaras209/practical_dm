import re
from pathlib import Path
from typing import Tuple


def highest_checkpoint(file_path: Path) -> Tuple[int, Path]:
    """
    Extract the checkpoint number from the given file path and return a tuple with the checkpoint number and
    the file path.

    Args:
        file_path (Path): The file path to a checkpoint file.

    Returns:
        Tuple[int, Path]: A tuple containing the checkpoint number as an integer and the original file path.
    """
    # Use regex to find a sequence of digits at the end of the file path string.
    digit_sequence = re.findall("\d+$", str(file_path))

    # If a sequence of digits is found, convert it to an integer.
    # Otherwise, set the checkpoint number to -1.
    checkpoint_number = int(digit_sequence[0]) if digit_sequence else -1

    # Return a tuple containing the checkpoint number and the original file path.
    return checkpoint_number, file_path
