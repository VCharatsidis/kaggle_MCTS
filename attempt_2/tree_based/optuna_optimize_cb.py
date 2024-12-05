import optuna
import catboost
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

features = best_features_5
categorical_features = train_pd[features].select_dtypes(include=['category']).columns.tolist()
print("Categorical features:", categorical_features)
print("features:", len(features))


def objective(trial):
    params = {
        'objective': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 200,
        'task_type' : 'GPU',  # Use GPU
        'devices' : '0',  # GPU device to use, default is '0'
        'verbose': 0,
        'early_stopping_rounds': 50,

        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'depth': trial.suggest_int('depth', 9, 13),
        #'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 500),
        # 'max_leaves': trial.suggest_int('max_leaves', 31, 255),
        # 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        # 'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        # 'rsm': trial.suggest_float('rsm', 0.6, 1.0),
        # 'border_count': trial.suggest_int('border_count', 32, 255),
        # 'grow_policy': 'Lossguide',
        # 'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),

    }

    scores = []
    test_ids = get_test_indexes_from_file('data_splits/test_set.txt')

    for fold in range(0, 5):
        fold_path = f'data_splits/Fold_{fold}.txt'
        fold_ids = get_test_indexes_from_file(fold_path)
        val_fold = train_pd[train_pd['Id'].isin(fold_ids)]
        train_fold = train_pd[(~train_pd['Id'].isin(fold_ids)) & (~train_pd['Id'].isin(test_ids))]

        X_train = train_fold[features].copy()
        y_train = train_fold['utility_agent1'].copy()

        X_val = val_fold[features].copy()
        y_val = val_fold['utility_agent1'].copy()

        model = catboost.CatBoostRegressor(**params, cat_features=categorical_features)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        scores.append(rmse)

    return np.mean(scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

print('Best params:', study.best_params)