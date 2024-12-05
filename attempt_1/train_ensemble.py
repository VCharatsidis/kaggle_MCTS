
import catboost
import lightgbm
import xgboost as xgb

import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error

from avoid_lists import avoid_features
from best_params import cb_params, xgb_params, lgbm_params, best_features
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

train_pd, _, _, _ = preprocess(train)

cat_features = train_pd[best_features].select_dtypes(include=['category']).columns.tolist()
for col in cat_features:
    train_pd[col] = train_pd[col].astype('category')

print("cat features:", cat_features)
print("features:", len(best_features))
print("to avoid:", len(avoid_features))

features_to_use = []
for c in train_pd[best_features].columns:
    num_unique = len(train_pd[best_features][c].unique())
    if num_unique < 200:
        features_to_use.append(c)
    print(c, num_unique)

best_features = features_to_use
print("features:", len(best_features))

test_ids = get_test_indexes_from_file('data_splits/test_set.txt')
test_data = train_pd[train_pd['Id'].isin(test_ids)]

X_test = test_data[best_features].copy()
y_test = test_data['utility_agent1'].copy()

train_fold = train_pd[(~train_pd['Id'].isin(test_ids))]
X_train = train_fold[best_features].copy()
y_train = train_fold['utility_agent1'].copy()

lgb_model = lightgbm.LGBMRegressor(**lgbm_params)
lgb_model.fit(X_train, y_train)
lgb_y_pred = lgb_model.predict(X_test)

cb_model = catboost.CatBoostRegressor(**cb_params, cat_features=cat_features)
cb_model.fit(X_train, y_train)
cb_y_pred = cb_model.predict(X_test)

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)

y_pred = (lgb_y_pred + xgb_y_pred + cb_y_pred) / 3

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("test rmse:", test_rmse)

RMSEs = []
for fold in range(1, 6):
    fold_path = f'attempt_1/data_splits/fold_{fold}.txt'
    fold_ids = get_test_indexes_from_file(fold_path)
    val_fold = train_pd[train_pd['Id'].isin(fold_ids)]
    train_fold = train_pd[(~train_pd['Id'].isin(fold_ids)) & (~train_pd['Id'].isin(test_ids))]

    X_train = train_fold[best_features].copy()
    y_train = train_fold['utility_agent1'].copy()

    X_val = val_fold[best_features].copy()
    y_val = val_fold['utility_agent1'].copy()

    lgb_model = lightgbm.LGBMRegressor(**lgbm_params)
    lgb_model.fit(X_train, y_train)
    lgb_y_pred = lgb_model.predict(X_val)

    cb_model = catboost.CatBoostRegressor(**cb_params, cat_features=cat_features)
    cb_model.fit(X_train, y_train)
    cb_y_pred = cb_model.predict(X_val)

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_y_pred = xgb_model.predict(X_val)

    y_pred = (lgb_y_pred + cb_y_pred + xgb_y_pred) / 3

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    RMSEs.append(rmse)

print(best_features)
print("Mean RMSE:", np.mean(RMSEs), "Std RMSE:", np.std(RMSEs), "Test RMSE:", test_rmse, "features:", len(best_features))

with open('ensemble_experiments.txt', 'a') as f:
    # Write the list of items, separated by commas
    f.write("Feature List: " + ', '.join(best_features) + '\n')
    f.write("Params lgb: " + ', '.join([f"{key}: {value}" for key, value in lgbm_params.items()]) + '\n')
    f.write('Params xgb: ' + ', '.join([f"{key}: {value}" for key, value in xgb_params.items()]) + '\n')
    f.write('Params cb: ' + ', '.join([f"{key}: {value}" for key, value in cb_params.items()]) + '\n')
    # Write the additional results like RMSE and feature count
    f.write(f"Mean RMSE: {np.mean(RMSEs):.4f}, Std RMSE: {np.std(RMSEs):.4f} ,Test RMSE: {test_rmse:.4f}, Features: {len(best_features)}\n")
    f.write('-' * 80 + '\n')  # Add a separator for readability
