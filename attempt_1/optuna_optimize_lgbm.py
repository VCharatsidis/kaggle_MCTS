import optuna
import xgboost
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error

from avoid_lists import avoid_features
from best_params import best_features
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

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.15, log=True),
        'n_estimators': 300,
        'max_depth': trial.suggest_int('max_depth', 7, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 40),
        'num_leaves': trial.suggest_int('num_leaves', 1500, 3000),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'verbose': -1
    }

    scores = []
    test_ids = get_test_indexes_from_file('data_splits/test_set.txt')

    for fold in range(1, 6):
        fold_path = f'attempt_1/data_splits/fold_{fold}.txt'
        fold_ids = get_test_indexes_from_file(fold_path)
        val_fold = train_pd[train_pd['Id'].isin(fold_ids)]
        train_fold = train_pd[(~train_pd['Id'].isin(fold_ids)) & (~train_pd['Id'].isin(test_ids))]

        X_train = train_fold[best_features].copy()
        y_train = train_fold['utility_agent1'].copy()

        X_val = val_fold[best_features].copy()
        y_val = val_fold['utility_agent1'].copy()

        model = xgboost.XGBRegressor(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        scores.append(rmse)

    return np.mean(scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

print('Best params:', study.best_params)