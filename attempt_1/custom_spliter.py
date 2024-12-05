import numpy as np
import polars as pl

def custom_split(df, num_splits):

    # Get value counts for 'column_a'
    value_counts = df["GameRulesetName"].value_counts()
    value_counts_dict = dict(zip(value_counts["GameRulesetName"], value_counts["count"]))
    games_rules_dict = dict(sorted(value_counts_dict.items(), key=lambda item: item[1], reverse=True))

    # for key in games_rules_dict.keys():
    #     print(f"{key}: {games_rules_dict[key]}")

    folds = [[] for _ in range(num_splits)]

    for gr in games_rules_dict.keys():
        gr_fights = df.filter(pl.col('GameRulesetName') == gr)
        unique_ids = list(gr_fights['Id'].unique())
        np.random.shuffle(unique_ids)

        lengths = [len(inner_list) for inner_list in folds]
        max_difference = max(lengths) - min(lengths)

        if max_difference > 100 or min(lengths) == 0:
            shortest_list_index = min(range(len(folds)), key=lambda i: len(folds[i]))
            folds[shortest_list_index].extend(unique_ids)
        else:
            means_utility = [df.filter(pl.col("Id").is_in(fold))['utility_agent1'].mean() for fold in folds]
            min_index = means_utility.index(min(means_utility))
            max_index = means_utility.index(max(means_utility))
            mean_utility_gr = gr_fights['utility_agent1'].mean()
            if mean_utility_gr < 0:
                folds[max_index].extend(unique_ids)
            else:
                folds[min_index].extend(unique_ids)


    for i, fold in enumerate(folds):

        print(f"Fold: {i}")
        print("length: ", len(fold))
        filtered_df = df.filter(pl.col("Id").is_in(fold))
        print("AdvantageP1 mean:",filtered_df['AdvantageP1'].mean())
        print("utility_agent1 mean:",filtered_df['utility_agent1'].mean())
        print()

    for fold in folds:
        filtered_df = df.filter(pl.col("Id").is_in(fold))
        mean_adv_p1 = round(filtered_df['AdvantageP1'].mean(), 3)
        mean_utility_agent1 = round(filtered_df['utility_agent1'].mean(), 3)

        with open(f"data_splits/mean_adv_p1_{mean_adv_p1}_utility_{mean_utility_agent1}.txt", "w") as file:
            # Convert the list to a string representation of integers separated by spaces
            numbers_str = " ".join(str(num) for num in fold)
            # Write the string to the file
            file.write(numbers_str)