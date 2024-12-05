import copy
import math

import numpy as np
import torch

from knn.knn_constants import EPOCHS, BATCH_SIZE_DEFAULT
from knn.knn_net import NeuralKNN
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from knn.simple_mlp import MLP


def train_fold(X_train, y_train, X_val, y_val, fold, embedding_dim, k, temperature):
    model = MLP(X_train.shape[1], embedding_dim, k=k, temperature=temperature).cuda()

    print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # DataLoader for mini-batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 1.5e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Track losses
    losses = []
    best_loss = float('inf')
    patience_counter = 20
    best_model = None

    # Number of rows to sample
    num_samples = 50000
    # pool_x = X_train[:num_samples].cuda()
    # pool_y = y_train[:num_samples].cuda()
    eval_every = 50

    for epoch in range(EPOCHS):
        model.train()

        epoch_loss = 0.0
        counter = 0
        for batch_x, batch_y in train_loader:
            # indices = torch.randperm(X_train.size(0))[:num_samples]
            #
            # sampled_tensor = X_train[indices]
            # sampled_labels = y_train[indices]
            #
            # model.set_support(sampled_tensor.cuda(), sampled_labels.cuda())

            # Move batch to GPU
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            preds_mean = model(batch_x).squeeze()

            # Calculate RMSE loss
            mse_loss = torch.mean((preds_mean - batch_y) ** 2)
            rmse = torch.sqrt(mse_loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            rmse.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += rmse.item()
            counter += 1


        # Learning rate scheduling
        #scheduler.step()

            # Validation and early stopping
            if counter % eval_every == 0:
                model.eval()

                val_rmse_sum = 0
                num_instances = 0
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.cuda(), val_y.cuda()

                        pred_mean = model(val_x).squeeze()
                        mse_loss = torch.sum((pred_mean - val_y) ** 2)
                        val_rmse_sum += mse_loss.item()
                        num_instances += val_x.shape[0]

                    val_rmse = math.sqrt(val_rmse_sum / num_instances)

                    model.train()
                    if val_rmse < best_loss:
                        best_loss = val_rmse
                        best_model = copy.deepcopy(model)
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    print(f'Epoch {epoch + 1}/{EPOCHS} - Train Loss: {epoch_loss/counter:.4f}, Val RMSE: {val_rmse:.4f}')
                    losses.append(val_rmse)

                    if patience_counter >= 10:  # Early stopping patience
                        print(f"Early stopping at epoch {epoch}")
                        break

    return best_model, val_rmse