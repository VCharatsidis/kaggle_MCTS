import optuna
import xgboost as xgb
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error

from avoid_lists import avoid_features
from best_params import best_features, xgb_params
from train_utils import preprocess, get_test_indexes_from_file

test = pl.read_csv('um-game-playing-strength-of-mcts-variants/test.csv')
print('Shape before dropping columns:', test.shape)

train = pl.read_csv('um-game-playing-strength-of-mcts-variants/train.csv')
print('Shape before dropping columns:', train.shape)

constant_columns = np.array(train.columns)[train.select(pl.all().n_unique() == 1).to_numpy().ravel()]
print(len(constant_columns), 'columns are constant. These will be dropped.')
drop_columns = list(constant_columns)
train = train.drop(drop_columns)
print('Shape after dropping columns:', train.shape)

train_pd, y, groups, cat_mapping = preprocess(train)

print("features:", len(best_features))
print("to avoid:", len(avoid_features))

# Identify categorical columns
cat_features = train_pd[best_features].select_dtypes(include=['category']).columns.tolist()
print("Categorical features:", cat_features)

# Convert categorical columns to 'category' dtype
for col in cat_features:
    train_pd[col] = train_pd[col].astype('category')

scores = []
test_ids = get_test_indexes_from_file('data_splits/test_set.txt')

test_data = train_pd[train_pd['Id'].isin(test_ids)]

X_test = test_data[best_features].copy()
y_test = test_data['utility_agent1'].copy()

train_fold = train_pd[(~train_pd['Id'].isin(test_ids))]
X_train = train_fold[best_features].copy()
y_train = train_fold['utility_agent1'].copy()

model = xgb.XGBRegressor(**xgb_params)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("test rmse:", rmse)

for fold in range(1, 6):
    fold_path = f'attempt_1/data_splits/fold_{fold}.txt'
    fold_ids = get_test_indexes_from_file(fold_path)
    val_fold = train_pd[train_pd['Id'].isin(fold_ids)]
    train_fold = train_pd[(~train_pd['Id'].isin(fold_ids)) & (~train_pd['Id'].isin(test_ids))]

    X_train = train_fold[best_features].copy()
    y_train = train_fold['utility_agent1'].copy()

    X_val = val_fold[best_features].copy()
    y_val = val_fold['utility_agent1'].copy()

    model = xgb.XGBRegressor(**xgb_params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    scores.append(rmse)

print("Mean RMSE:", np.mean(scores))



