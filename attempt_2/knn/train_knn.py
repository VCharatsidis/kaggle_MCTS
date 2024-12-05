import math

import catboost
import lightgbm
import pandas as pd
import torch
import xgboost as xgb

import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from attempt_2.tree_based.best_params import best_features_5
from attempt_2.tree_based.train_utils import preprocess, get_test_indexes_from_file
from knn.knn_constants import BATCH_SIZE_DEFAULT
from knn.train_a_fold import train_fold


train = pl.read_csv('../../um-game-playing-strength-of-mcts-variants/train.csv')
print('Shape before dropping columns:', train.shape)

# for c in train.columns:
#     print(c, len(train[c].unique()))


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

# features = [f for f in features if len(train_pd[f].unique()) < 200]

print("cat features:", cat_features)
print(features)
print("features:", len(features))

test_ids = get_test_indexes_from_file('../data_splits/test_set.txt')
test_data = train_pd[train_pd['Id'].isin(test_ids)]

X_test = test_data[features].copy()
y_test = test_data['utility_agent1'].copy()

train_data_fold = train_pd[(~train_pd['Id'].isin(test_ids))]
X_train = train_data_fold[features].copy()
y_train = train_data_fold['utility_agent1'].copy()


mx = X_train.mean()
std = X_train.std()

X_test = (X_test - mx) / std


embedding_dim = 8
k = 1000
temperature = 0.35

X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False)


RMSEs = []
test_RMSEs = []
for fold in range(0, 5):
    fold_path = f'../data_splits/fold_{fold}.txt'
    fold_ids = get_test_indexes_from_file(fold_path)
    val_fold = train_pd[train_pd['Id'].isin(fold_ids)]
    train_data_fold = train_pd[(~train_pd['Id'].isin(fold_ids)) & (~train_pd['Id'].isin(test_ids))]

    X_train = train_data_fold[features].copy()
    y_train = train_data_fold['utility_agent1'].copy()

    X_val = val_fold[features].copy()
    y_val = val_fold['utility_agent1'].copy()

    eps = 0.0001
    X_train = (X_train - mx) / (std + eps)
    X_val = (X_val - mx) / (std + eps)

    train_x = torch.tensor(X_train.values, dtype=torch.float32)
    train_y = torch.tensor(y_train.values, dtype=torch.float32)
    val_x = torch.tensor(X_val.values, dtype=torch.float32)
    val_y = torch.tensor(y_val.values, dtype=torch.float32)

    print("train_x shape:", train_x.shape, "train_y shape:", train_y.shape)

    model, val_rmse = train_fold(train_x, train_y, val_x, val_y, fold, embedding_dim, k, temperature)
    torch.save(model, f"knn_models/knn_fold_{fold}_emb_{embedding_dim}_k_{k}_temp_{temperature}.model")
    RMSEs.append(val_rmse)

    test_rmse_sum = 0
    num_instances = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.cuda(), test_y.cuda()

            pred_mean = model(test_x).squeeze()
            mse_loss = torch.sum((pred_mean - test_y) ** 2)
            test_rmse_sum += mse_loss.item()
            num_instances += test_y.shape[0]

    test_rmse = math.sqrt(test_rmse_sum / num_instances)

    test_RMSEs.append(test_rmse)

print(features)
print("Mean RMSE:", np.mean(RMSEs), "Std RMSE:", np.std(RMSEs), "Test RMSE:", np.mean(test_RMSEs), "features:", len(features))

with open('knn_experiments.txt', 'a') as f:
    # Write the list of items, separated by commas
    f.write("Feature List: " + ', '.join(features) + '\n')
    # Write the additional results like RMSE and feature count
    f.write(f"Mean RMSE: {np.mean(RMSEs):.4f}, Std RMSE: {np.std(RMSEs):.4f} ,Test RMSE: {np.mean(test_RMSEs):.4f}, Features: {len(features)}\n")
    f.write('-' * 80 + '\n')  # Add a separator for readability

