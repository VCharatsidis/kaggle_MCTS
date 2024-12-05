import os

import polars as pl
import pandas as pd


def get_test_indexes_from_file(fold_path):
    with open(fold_path, "r") as file:
        # Read the contents of the file
        file_contents = file.read()
        # Convert the string back to a list of integers
        test_set_indexes = [id for id in file_contents.split()]

    test_set_indexes = [int(x) for x in test_set_indexes]
    return test_set_indexes


def preprocess(df_polars):
    """Convert the polars dataframe to pandas; extract target and groups if it is the training dataframe

    The function should be applied to training and test datasets.

    Parameters
    df_polars: polars DataFrame (train or test)

    Return values:
    df: pandas DataFrame with all features of shape (n_samples, n_features)
    target: target array of shape (n_samples, ) or None
    groups: grouping array for GroupKFold of shape (n_samples, ) or None
    """

    df_polars = df_polars.with_columns(pl.col("EnglishRules").str.len_chars().alias("char_count_english"))
    df_polars = df_polars.with_columns(pl.col("LudRules").str.len_chars().alias("char_count_lud"))

    # Add eight features extracted from player names,
    # Drop GameRulesetName, freetext and target columns
    df = df_polars.with_columns(
        pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p1_selection'),
        pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p1_exploration').cast(pl.Float32),
        pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p1_playout'),
        pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p1_bounds'),
        pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p2_selection'),
        pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p2_exploration').cast(pl.Float32),
        pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p2_playout'),
        pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p2_bounds')
    ).drop(
        ['GameRulesetName', 'EnglishRules', 'LudRules']
    ).to_pandas()

    if 'utility_agent1' in df_polars.columns:  # Processing the training data
        # Extract the target
        target = df_polars.select('utility_agent1').to_numpy().ravel()

        # Extract the groups for the GroupKFold
        groups = df_polars.select('GameRulesetName').to_numpy()

        # Set the mapping to categorical dtypes
        cat_mapping = {feature: pd.CategoricalDtype(categories=list(set(df[feature]))) for feature in
                       df.columns[df.dtypes == object]}
    else:  # Processing the test data
        target, groups = None, None

    # Convert the strings to categorical
    df = df.astype(cat_mapping)

    return df, target, groups, cat_mapping


