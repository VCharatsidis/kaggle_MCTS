
import catboost
import lightgbm
import pandas as pd
import torch
import xgboost as xgb

import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error

from attempt_2.gps.train_gp import train_a_gp, predict_inference, train_a_gp_batched, train_svgp_with_batches
from attempt_2.tree_based.best_params import best_features_5, lgb_params_5, xgb_params_5, cb_params_5
from attempt_2.tree_based.train_utils import preprocess, get_test_indexes_from_file

test = pl.read_csv('../../um-game-playing-strength-of-mcts-variants/test.csv')
print('Shape before dropping columns:', test.shape)

train = pl.read_csv('../../um-game-playing-strength-of-mcts-variants/train.csv')
print('Shape before dropping columns:', train.shape)

constant_columns = np.array(train.columns)[train.select(pl.all().n_unique() == 1).to_numpy().ravel()]
print(len(constant_columns), 'columns are constant. These will be dropped.')
drop_columns = list(constant_columns)
train = train.drop(drop_columns)
print('Shape after dropping columns:', train.shape)

features = best_features_5
train_pd, _, _, _ = preprocess(train)

cat_features = train_pd[features].select_dtypes(include=['category']).columns.tolist()
for col in cat_features:
    train_pd[col] = train_pd[col].astype('category')

original_cols = train_pd.columns.tolist()
train_pd = pd.get_dummies(train_pd, columns=cat_features)
updated_columns = train_pd.columns.tolist()

new_cols = list(set(updated_columns) - set(original_cols))
features = [f for f in features if f not in cat_features] + new_cols

print("cat features:", cat_features)
print(features)
print("features:", len(features))

test_ids = get_test_indexes_from_file('../data_splits/test_set.txt')
test_data = train_pd[train_pd['Id'].isin(test_ids)]

X_test = test_data[features].copy()
y_test = test_data['utility_agent1'].copy()

train_fold = train_pd[(~train_pd['Id'].isin(test_ids))]
X_train = train_fold[features].copy()
y_train = train_fold['utility_agent1'].copy()

mx = X_train.mean()
std = X_train.std()
X_train = (X_train - mx) / std
X_test = (X_test - mx) / std

X_test = torch.tensor(X_test.values, dtype=torch.float32).cuda()
y_test = torch.tensor(y_test.values, dtype=torch.float32).cuda()

RMSEs = []
test_RMSEs = []
for fold in range(0, 5):
    fold_path = f'../data_splits/fold_{fold}.txt'
    fold_ids = get_test_indexes_from_file(fold_path)
    val_fold = train_pd[train_pd['Id'].isin(fold_ids)]
    train_fold = train_pd[(~train_pd['Id'].isin(fold_ids)) & (~train_pd['Id'].isin(test_ids))]

    X_train = train_fold[features].copy()
    y_train = train_fold['utility_agent1'].copy()

    X_val = val_fold[features].copy()
    y_val = val_fold['utility_agent1'].copy()

    train_x = torch.tensor(X_train.values, dtype=torch.float32)
    train_y = torch.tensor(y_train.values, dtype=torch.float32)
    val_x = torch.tensor(X_val.values, dtype=torch.float32)
    val_y = torch.tensor(y_val.values, dtype=torch.float32)

    print("train_x shape:", train_x.shape, "train_y shape:", train_y.shape)

    # Train model
    model, likelihood, losses = train_a_gp(train_x, train_y, val_x, val_y, n_epochs=100)

    pred_mean = predict_inference(model, val_x)
    mse_loss = torch.mean((pred_mean - val_y) ** 2)
    rmse = torch.sqrt(mse_loss)
    RMSEs.append(rmse)

    pred_mean = predict_inference(model, X_test)
    mse_loss = torch.mean((pred_mean - y_test) ** 2)
    rmse = torch.sqrt(mse_loss)
    test_RMSEs.append(rmse)

print(features)
print("Mean RMSE:", np.mean(RMSEs), "Std RMSE:", np.std(RMSEs), "Test RMSE:", np.mean(test_RMSEs), "features:", len(features))

with open('ensemble_experiments.txt', 'a') as f:
    # Write the list of items, separated by commas
    f.write("Feature List: " + ', '.join(features) + '\n')
    f.write("Params lgb: " + ', '.join([f"{key}: {value}" for key, value in lgb_params_5.items()]) + '\n')
    f.write('Params xgb: ' + ', '.join([f"{key}: {value}" for key, value in xgb_params_5.items()]) + '\n')
    f.write('Params cb: ' + ', '.join([f"{key}: {value}" for key, value in cb_params_5.items()]) + '\n')
    # Write the additional results like RMSE and feature count
    f.write(f"Mean RMSE: {np.mean(RMSEs):.4f}, Std RMSE: {np.std(RMSEs):.4f} ,Test RMSE: {np.mean(test_RMSEs):.4f}, Features: {len(features)}\n")
    f.write('-' * 80 + '\n')  # Add a separator for readability

