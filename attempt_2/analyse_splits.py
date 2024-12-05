import polars as pl

def get_test_indexes_from_file(fold_path):
    with open(fold_path, "r") as file:
        # Read the contents of the file
        file_contents = file.read()
        # Convert the string back to a list of integers
        test_set_indexes = [id for id in file_contents.split()]

    test_set_indexes = [int(x) for x in test_set_indexes]
    return test_set_indexes


test_ids = get_test_indexes_from_file('data_splits/test_set.txt')

fold_1 = get_test_indexes_from_file('data_splits/fold_1.txt')
fold_2 = get_test_indexes_from_file('data_splits/fold_2.txt')
fold_3 = get_test_indexes_from_file('data_splits/fold_3.txt')
fold_4 = get_test_indexes_from_file('data_splits/fold_4.txt')
fold_5 = get_test_indexes_from_file('data_splits/test_set.txt')

train = pl.read_csv('../um-game-playing-strength-of-mcts-variants/train.csv')
print(len(train['GameRulesetName'].unique()))
print('Shape before dropping columns:', train.shape)

filtered_df_1 = train.filter(pl.col('Id').is_in(fold_1))
print(filtered_df_1.shape)

filtered_df_2 = train.filter(pl.col('Id').is_in(fold_3))
print(filtered_df_2.shape)

game_rules_1 = list(filtered_df_1['GameRulesetName'].unique())
game_rules_2 = list(filtered_df_2['GameRulesetName'].unique())

comon_ids = [x for x in game_rules_2 if x in game_rules_1]
print(game_rules_1, print(len(game_rules_1)))
print(game_rules_2, print(len(game_rules_2)))
print(comon_ids)