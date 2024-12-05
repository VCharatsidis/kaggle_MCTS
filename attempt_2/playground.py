import polars as pl

test = pl.read_csv('../um-game-playing-strength-of-mcts-variants/test.csv')
print('Shape before dropping columns:', test.shape)

train = pl.read_csv('../um-game-playing-strength-of-mcts-variants/train.csv')
print('Shape before dropping columns:', train.shape)

train = train.with_columns(pl.col("EnglishRules").str.len_chars().alias("char_count_english"))
train = train.with_columns(pl.col("LudRules").str.len_chars().alias("char_count_lud"))

print(train['char_count_english'].mean())
print(train['char_count_lud'].mean())

print(train['char_count_english'].std())
print(train['char_count_lud'].std())

print(train['char_count_english'].max(), train['char_count_english'].min())
print(train['char_count_lud'].max(), train['char_count_lud'].min())