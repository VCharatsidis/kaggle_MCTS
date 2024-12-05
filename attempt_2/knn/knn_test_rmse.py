import math

import pandas as pd
import torch


import polars as pl
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from attempt_2.tree_based.best_params import best_features_5
from attempt_2.tree_based.train_utils import preprocess, get_test_indexes_from_file
from knn.knn_constants import BATCH_SIZE_DEFAULT


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

train_data_fold = train_pd[(~train_pd['Id'].isin(test_ids))]
X_train = train_data_fold[features].copy()

mx = X_train.mean()
std = X_train.std()

X_test = (X_test - mx) / std

embedding_dim = 128
k = 1000
temperature = 0.75

X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False)


def cal_test_rmse():
    test_RMSEs = []
    models = []
    for fold in range(0, 5):
        model = torch.load(f"knn_models/knn_fold_{fold}_emb_{embedding_dim}_k_{k}_temp_{temperature}.model")
        model.eval()
        models.append(model)

    test_rmse_sum = 0
    num_instances = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.cuda(), test_y.cuda()

            preds = []
            for model in models:
                pred_mean = model(test_x).squeeze()
                preds.append(pred_mean)

            pred_mean = torch.stack(preds, dim=0).mean(dim=0)
            mse = (pred_mean - test_y) ** 2
            mse_loss = torch.sum(mse)
            test_rmse_sum += mse_loss.item()
            num_instances += test_y.shape[0]

    test_rmse = math.sqrt(test_rmse_sum / num_instances)

    test_RMSEs.append(test_rmse)

    return test_RMSEs


test_RMSEs = cal_test_rmse()
print(test_RMSEs)