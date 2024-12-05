import optuna
import lightgbm
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error

from attempt_2.tree_based.best_params import best_features_5
from train_utils import preprocess, get_test_indexes_from_file

test = pl.read_csv('../um-game-playing-strength-of-mcts-variants/test.csv')
print('Shape before dropping columns:', test.shape)

train = pl.read_csv('../um-game-playing-strength-of-mcts-variants/train.csv')
print('Shape before dropping columns:', train.shape)

constant_columns = np.array(train.columns)[train.select(pl.all().n_unique() == 1).to_numpy().ravel()]
print(len(constant_columns), 'columns are constant. These will be dropped.')
drop_columns = list(constant_columns)
train = train.drop(drop_columns)
print('Shape after dropping columns:', train.shape)

train_pd, y, groups, cat_mapping = preprocess(train)

# Identify categorical columns
# cat_features = train_pd[best_features].select_dtypes(include=['category']).columns.tolist()
# print("Categorical features:", cat_features)
#
# # Convert categorical columns to 'category' dtype
# for col in cat_features:
#     train_pd[col] = train_pd[col].astype('category')

features_to_use = best_features_5
print("features:", len(features_to_use))

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'verbose': -1,

        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 7, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 40),
        'num_leaves': trial.suggest_int('num_leaves', 1500, 3000),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10)

    }

    scores = []
    test_ids = get_test_indexes_from_file('data_splits/test_set.txt')

    for fold in range(0, 5):
        fold_path = f'data_splits/fold_{fold}.txt'
        fold_ids = get_test_indexes_from_file(fold_path)
        val_fold = train_pd[train_pd['Id'].isin(fold_ids)]
        train_fold = train_pd[(~train_pd['Id'].isin(fold_ids)) & (~train_pd['Id'].isin(test_ids))]

        X_train = train_fold[features_to_use].copy()
        y_train = train_fold['utility_agent1'].copy()

        X_val = val_fold[features_to_use].copy()
        y_val = val_fold['utility_agent1'].copy()

        model = lightgbm.LGBMRegressor(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        scores.append(rmse)

    return np.mean(scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

print('Best params:', study.best_params)