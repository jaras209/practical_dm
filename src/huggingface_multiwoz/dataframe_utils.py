import pandas as pd
import numpy as np
import random


def print_examples(df_orig, df_subset, num_examples=2):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)
    # Get the list of dialogue_ids that are in both df_orig and df_subset
    common_dialogue_ids = set(df_orig['dialogue_id']).intersection(set(df_subset['dialogue_id']))

    # If there are fewer common dialogues than requested num_examples, print a warning and adjust num_examples
    if len(common_dialogue_ids) < num_examples:
        print(f"Warning: Only {len(common_dialogue_ids)} common dialogues found. Adjusting num_examples.")
        num_examples = len(common_dialogue_ids)

    # Select num_examples dialogue_ids randomly from common_dialogue_ids
    example_dialogue_ids = random.sample(common_dialogue_ids, num_examples)

    # For each selected dialogue_id, print the rows in df_orig and df_subset that correspond to this dialogue_id
    for dialogue_id in example_dialogue_ids:
        print(f"Dialogue ID: {dialogue_id}")
        print("Rows in original DataFrame:")
        print(df_orig[df_orig['dialogue_id'] == dialogue_id])
        print("\nRows in subset DataFrame:")
        print(df_subset[df_subset['dialogue_id'] == dialogue_id])
        print("\n---\n")


def print_df_statistics(df: pd.DataFrame, df_subset: pd.DataFrame) -> None:
    """
    Print statistics about the original and subset dataframes.
    Args:
        df (pd.DataFrame): The original dataframe.
        df_subset (pd.DataFrame): The subset dataframe.
    """
    print("Original dataframe:")
    print(f"Total dialogues: {df['dialogue_id'].nunique()}")
    print(f"Total turns: {df.shape[0]}")
    print(df.groupby('domain')['dialogue_id'].nunique())
    print("\n")

    print("Subset dataframe:")
    print(f"Total dialogues: {df_subset['dialogue_id'].nunique()}")
    print(f"Total turns: {df_subset.shape[0]}")
    print(df_subset.groupby('domain')['dialogue_id'].nunique())
    print("\n")

    print("Portions:")
    print(f"Dialogues: {df_subset['dialogue_id'].nunique() / df['dialogue_id'].nunique()}")
    print(f"Turns: {df_subset.shape[0] / df.shape[0]}")
    print(df_subset.groupby('domain')['dialogue_id'].nunique() / df.groupby('domain')['dialogue_id'].nunique())


def get_dialogue_subset(df: pd.DataFrame, subset_ratio: float, random_seed: int = 42) -> pd.DataFrame:
    """
    Get a subset of the dataframe based on domain and subset_ratio.
    Args:
        df (pd.DataFrame): The dataframe from which to get the subset.
        subset_ratio (float): The fraction of the dataframe to take as a subset. Should be between 0 and 1.
        random_seed (int): The seed for the random number generator. Defaults to 42.
    Returns:
        pd.DataFrame: The subset dataframe.
    """
    # Assert that subset_ratio is between 0 and 1
    assert 0 <= subset_ratio <= 1, "subset_ratio should be between 0 and 1"

    # Group the dataframe by 'domain' and 'dialogue_id', and then sample the unique dialogue ids
    unique_dialogue_ids = df.groupby(['domain', 'dialogue_id']).ngroup()
    dialogue_id_df = pd.DataFrame({'dialogue_id': df['dialogue_id'], 'unique_dialogue_id': unique_dialogue_ids,
                                   'domain': df['domain']}).drop_duplicates()

    # Sample the dialogues from each domain
    sampled_dialogue_ids = dialogue_id_df.groupby('domain').apply(
        lambda x: x.sample(frac=subset_ratio, random_state=random_seed))['unique_dialogue_id']

    # Use the sampled dialogue ids to get the subset of the original dataframe
    df_subset = df[df['dialogue_id'].isin(sampled_dialogue_ids)]

    return df_subset
