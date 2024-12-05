import polars as pl
import numpy as np

train = pl.read_csv('um-game-playing-strength-of-mcts-variants/train.csv')
print('Shape before dropping columns:', train.shape)

constant_columns = np.array(train.columns)[train.select(pl.all().n_unique() == 1).to_numpy().ravel()]
print(len(constant_columns), 'columns are constant. These will be dropped.')

id_columns = np.array(train.columns)[train.select(pl.all().n_unique() == train.shape[0]).to_numpy().ravel()]
print('There are', len(id_columns), 'id-like columns:', id_columns)

drop_columns = list(constant_columns) + ['Id']

train = train.drop(drop_columns)
print('Shape after dropping columns:', train.shape)

# Null values
print('There are', train.null_count().to_numpy().sum(), 'missing values.')

# Duplicates
print('There are', len(train) - train.n_unique(), 'duplicates.')

# Boolean columns
print('There are', train.select(pl.all().n_unique() == 2).to_numpy().sum(), 'binary columns.')

games_played_info = train.select(pl.col('^num_.*$')).sum_horizontal().rename('num_games_played').value_counts().sort('num_games_played')
print(games_played_info)

end_in_draw = train.select((pl.col('num_wins_agent1') + pl.col('num_losses_agent1') == 0).alias('Games which always end in a draw')).sum()
print('There are', end_in_draw, 'games which always end in a draw.')


# Advantage for the player who makes the first move (average over all games)
print(train.select('utility_agent1').mean())

print(train.select(pl.col('GameRulesetName')).to_series().value_counts(sort=True))

print("Groupwise standard deviations of columns")
print(train.select('GameRulesetName', pl.col(pl.Int64), pl.col(pl.Float64))
      .group_by('GameRulesetName')
      .agg(pl.all().std())
      .drop('GameRulesetName')
      .max()
      .transpose(include_header=True, header_name='Feature', column_names=['max std'])
      .sort('max std')
     )

print(train.select(pl.col('agent1')).to_series().value_counts(sort=True))