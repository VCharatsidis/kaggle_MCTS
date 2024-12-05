import math
import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error  # or your relevant metric
import polars as pl
import pandas as pd
from best_params import best_features_5
from knn.knn_constants import BATCH_SIZE_DEFAULT
from train_utils import get_test_indexes_from_file, preprocess
from torch.utils.data import DataLoader, TensorDataset


def do_eval(val_loader, model):
    val_rmse_sum = 0
    num_instances = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            #val_x, val_y = val_x.cuda(), val_y.cuda()

            pred_mean = model(val_x).squeeze()
            mse_loss = torch.sum((pred_mean - val_y) ** 2)
            val_rmse_sum += mse_loss.item()
            num_instances += val_x.shape[0]

        val_rmse = math.sqrt(val_rmse_sum / num_instances)

    return val_rmse


def permutation_importance(model, X, y, feature_names, n_repeats):
    """
    Calculate permutation importance for a neural network

    Parameters:
    - model: Trained PyTorch model
    - X_val: Validation features (numpy array or torch tensor)
    - y_val: Validation targets (numpy array or torch tensor)
    - metric: Performance metric to use (default is MSE)
    - n_repeats: Number of permutation shuffles

    Returns:
    - Dictionary of feature importances
    """

    val_dataset = TensorDataset(X, y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False)

    # Get baseline performance
    model.eval()

    val_rmse = do_eval(val_loader, model)

    # Store feature importances
    importances = {}

    # Iterate through features
    for feature_idx in range(X.shape[1]):
        feature_importances = []

        # Repeat permutation multiple times

        for _ in range(n_repeats):
            # Create a copy of validation data
            X_permuted = X.clone()

            # Shuffle the specific feature
            permutation = torch.randperm(X.shape[0])
            X_permuted[:, feature_idx] = X[permutation, feature_idx]

            permuted_dataset = TensorDataset(X_permuted, y)
            permute_loader = DataLoader(permuted_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False)

            permuted_rmse = do_eval(permute_loader, model)
            print(f'Permutation {feature_idx} permuted_rmse: {permuted_rmse}', "val_rmse:", val_rmse)

            # Calculate importance as performance drop
            feature_importances.append(val_rmse - permuted_rmse)

        # Store mean importance for this feature
        importances[f'feature_{feature_idx}'] = np.mean(feature_importances)
        print(f'Feature {feature_idx}, {feature_names[feature_idx]} importance: {importances[f"feature_{feature_idx}"]}')

    return importances


def feature_importance():

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
    y_train = train_data_fold['utility_agent1'].copy()

    mx = X_train.mean()
    std = X_train.std()

    X_test = (X_test - mx) / std

    for fold in range(0, 5):
        model = torch.load(f"knn_models/knn_fold_{fold}_emb_64_k_1000_temp_0.35.model")
        model.cuda()

        fold_path = f'../data_splits/fold_{fold}.txt'
        fold_ids = get_test_indexes_from_file(fold_path)
        val_fold = train_pd[train_pd['Id'].isin(fold_ids)]
        train_data_fold = train_pd[(~train_pd['Id'].isin(fold_ids)) & (~train_pd['Id'].isin(test_ids))]

        X_train = train_data_fold[features].copy()
        y_train = train_data_fold['utility_agent1'].copy()

        X_val = val_fold[features].copy()
        y_val = val_fold['utility_agent1'].copy()

        X_train = (X_train - mx) / std
        X_val = (X_val - mx) / std

        val_x = torch.tensor(X_train.values, dtype=torch.float32).cuda()
        val_y = torch.tensor(y_train.values, dtype=torch.float32).cuda()

        importances = permutation_importance(model, val_x, val_y, features, n_repeats=2)
        print(importances)
        print("============")
        print()


feature_importance()
