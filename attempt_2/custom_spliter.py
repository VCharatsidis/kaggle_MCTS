import numpy as np
import polars as pl

def custom_split(df, num_splits):

    # Get value counts for 'column_a'
    value_counts = df["GameRulesetName"].value_counts()
    value_counts_dict = dict(zip(value_counts["GameRulesetName"], value_counts["count"]))
    games_rules_dict = dict(sorted(value_counts_dict.items(), key=lambda item: item[1], reverse=True))

    folds = [[] for _ in range(num_splits)]

    for gr in games_rules_dict.keys():
        gr_fights = df.filter(pl.col('GameRulesetName') == gr)
        unique_ids = list(gr_fights['Id'].unique())
        np.random.shuffle(unique_ids)

        shortest_list_index = min(range(len(folds)), key=lambda i: len(folds[i]))
        folds[shortest_list_index].extend(unique_ids)

    for i, fold in enumerate(folds):

        print(f"Fold: {i}")
        print("length: ", len(fold))
        filtered_df = df.filter(pl.col("Id").is_in(fold))
        print("AdvantageP1 mean:",filtered_df['AdvantageP1'].mean())
        print("utility_agent1 mean:",filtered_df['utility_agent1'].mean())
        print()

        with open(f"data_splits/Fold_{i}.txt", "w") as file:
            # Convert the list to a string representation of integers separated by spaces
            numbers_str = " ".join(str(num) for num in fold)
            # Write the string to the file
            file.write(numbers_str)