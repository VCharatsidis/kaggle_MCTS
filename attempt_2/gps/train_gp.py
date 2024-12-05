import copy

import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        # Variational strategy with inducing points
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        )
        super().__init__(variational_strategy)


        # Mean and kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = torch.tensor(1.0)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def predict_inference(model, test_x):
    """Make predictions with uncertainty estimates"""
    model.eval()

    with torch.no_grad():
        # Get output distribution
        output = model(test_x)
        # Transform to probabilities
        mean = output.mean

    return mean


class StandardGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(StandardGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x.size(1))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

        # Initialize parameters more sensibly
        self.covar_module.outputscale = 1.0
        if hasattr(self.covar_module.base_kernel, 'lengthscale'):
            self.covar_module.base_kernel.lengthscale = 1.0

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_a_gp(train_x, train_y, val_x, val_y, n_epochs=100, learning_rate=0.01):

    """Train GP using BCE loss with improved training process"""
    # Initialize model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = StandardGPModel(train_x, train_y, likelihood)

    print("model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Training mode
    model.train()
    likelihood.train()

    # Use Adam optimizer with weight decay
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': learning_rate}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_model = 0

    for i in range(n_epochs):
        model.train()
        likelihood.train()
        optimizer.zero_grad()

        # Get GP output distribution
        output = model(train_x)
        preds_mean = output.mean

        # Calculate rmse loss

        mse_loss = torch.mean((preds_mean - train_y) ** 2)
        rmse = torch.sqrt(mse_loss)

        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())
        total_loss = rmse + 0.01 * l2_reg

        # Backward pass and optimization
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Learning rate scheduling
        scheduler.step()

        if (i + 1) % 1 == 0:
            with torch.no_grad():
                # Get GP output distribution
                pred_mean = predict_inference(model, val_x)

                mse_loss = torch.mean((pred_mean - val_y) ** 2)
                rmse = torch.sqrt(mse_loss)

                if rmse.item() < best_loss:
                    best_loss = rmse.item()
                    best_model = copy.deepcopy(model)
                    patience_counter = 0

                if patience_counter >= 100:  # Early stopping patience
                    print(f"Early stopping at epoch {i}")
                    break

                patience_counter += 1
                print(f'Epoch {i + 1}/{n_epochs} - Loss: {rmse.item():.4f}')
                # Print predictions for a few points to monitor learning

    return best_model, likelihood, losses


class BatchedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_size):
        super().__init__(train_x, train_y, likelihood)
        self.batch_size = batch_size

        # Define batched kernel and mean modules
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch_size]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([batch_size])),
            batch_shape=torch.Size([batch_size])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def predict_inference_batched(model, test_x):
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = model(test_x)
        return output.mean

def train_a_gp_batched(train_x, train_y, val_x, val_y, n_epochs=100, learning_rate=0.01, batch_size=512):
    """
    Train a GP using mini-batch training.
    """
    import gpytorch
    import torch
    import copy
    from torch.utils.data import DataLoader, TensorDataset

    # Initialize model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = SVGPModel(train_x, train_y, likelihood).cuda()

    print("Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # DataLoader for mini-batch processing
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': learning_rate}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # Track losses
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_model = None

    for epoch in range(n_epochs):
        model.train()
        likelihood.train()

        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Move batch to GPU
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            # Get GP output distribution
            output = model(batch_x)
            preds_mean = output.mean

            # Calculate RMSE loss
            mse_loss = torch.mean((preds_mean - batch_y) ** 2)
            rmse = torch.sqrt(mse_loss)

            # Add L2 regularization
            l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())
            total_loss = rmse + 0.01 * l2_reg

            # Backward pass and optimization
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        # Learning rate scheduling
        scheduler.step()

        # Validation and early stopping
        if (epoch + 1) % 10 == 0:
            model.eval()
            likelihood.eval()

            with torch.no_grad():
                pred_mean = predict_inference(model, val_x.cuda())
                mse_loss = torch.mean((pred_mean - val_y.cuda()) ** 2)
                val_rmse = torch.sqrt(mse_loss)

                if val_rmse.item() < best_loss:
                    best_loss = val_rmse.item()
                    best_model = copy.deepcopy(model)
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f'Epoch {epoch + 1}/{n_epochs} - Train Loss: {epoch_loss:.4f}, Val RMSE: {val_rmse.item():.4f}')
                losses.append(val_rmse.item())

                if patience_counter >= 10:  # Early stopping patience
                    print(f"Early stopping at epoch {epoch}")
                    break

    return best_model, likelihood, losses


def assert_no_nan(tensor, name="Tensor"):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values!"


def train_svgp_with_batches(train_x, train_y, val_x, val_y, n_epochs=100, learning_rate=0.01, batch_size=512):
    # Define inducing points (random subset of training data)
    inducing_points = train_x[:4000]  # Choose 50 inducing points (can be tuned)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = SVGPModel(inducing_points=inducing_points).cuda()

    # Use Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)

    # DataLoader for mini-batches
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        likelihood.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            optimizer.zero_grad()
            # Get GP output distribution
            output = model(batch_x)

            preds_mean = output.mean

            # Calculate RMSE loss
            mse_loss = torch.mean((preds_mean - batch_y) ** 2)
            rmse = torch.sqrt(mse_loss)

            rmse.backward()
            optimizer.step()

            epoch_loss += rmse.item()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

    return model, likelihood