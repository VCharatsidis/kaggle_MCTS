import time

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from attempt_2.tree_based.train_utils import get_test_indexes_from_file
from transformers import LongformerModel, LongformerTokenizer

class TextRegressionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.float)
        }


class TextRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name='distilbert-base-uncased', dropout_rate=0.1):
        super(TextRegressionModel, self).__init__()
        self.bert = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        self.activation = nn.Tanh()  # To ensure output between -1 and 1

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        return self.activation(logits)


def prepare_data(texts, labels, batch_size=2, max_length=4096):
    """
    Prepare data for training/evaluation.

    Args:
        texts (list): List of input texts
        labels (list): List of numerical labels
        batch_size (int): Batch size for DataLoader
        max_length (int): Maximum sequence length

    Returns:
        tokenizer: The initialized tokenizer
        data_loader: DataLoader containing the processed dataset
    """
    # Initialize tokenizer
    #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    # Create dataset
    dataset = TextRegressionDataset(texts, labels, tokenizer, max_length)

    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return tokenizer, data_loader


def train_epoch_fast(model, train_loader, val_loader, learning_rate=1e-5):
    from torch.cuda.amp import autocast, GradScaler
    model.train()
    total_loss = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model = model.cuda()

    best_val_loss = float('inf')

    # Use gradient scaler for mixed precision training
    scaler = GradScaler()
    counter = 0
    # Disable gradient synchronization for faster training
    torch.backends.cudnn.benchmark = True
    eval_every = 2000
    n_update = 16

    optimizer.zero_grad()
    accumulated_loss = 0

    for batch in train_loader:
        start_time = time.time()
        # Entire batch to device at once
        input_ids = batch['input_ids'].to('cuda', non_blocking=True)
        attention_mask = batch['attention_mask'].to('cuda', non_blocking=True)
        targets = batch['target'].to('cuda', non_blocking=True)

        # Use automatic mixed precision
        with autocast():
            outputs = model(input_ids, attention_mask)

            # More efficient loss calculation
            loss = torch.nn.functional.mse_loss(outputs.squeeze(), targets, reduction='mean')
            loss = torch.sqrt(loss)

        # Scaled gradient descent
        scaler.scale(loss).backward()

        if (counter+1) % n_update == 0:
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()

        counter+=1
        #print(f'Batch to device time: {time.time() - start_time:.4f}')

        if counter % eval_every == 0:

            # Validation
            model.eval()
            val_loss = 0
            val_counter = 0
            with torch.no_grad():
                for batch in val_loader:
                    if val_counter > eval_every:
                        break
                    input_ids = batch['input_ids'].cuda(non_blocking=True)
                    attention_mask = batch['attention_mask'].cuda(non_blocking=True)
                    targets = batch['target'].cuda(non_blocking=True)

                    with autocast():
                        outputs = model(input_ids, attention_mask)

                        # More efficient loss calculation
                        loss = torch.nn.functional.mse_loss(outputs.squeeze(), targets, reduction='mean')
                        loss = torch.sqrt(loss)

                    val_loss += loss.item()
                    val_counter += 1

            avg_train_loss = total_loss / eval_every
            total_loss = 0
            avg_val_loss = val_loss / eval_every

            print(f'Iter {counter}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, 'best_model.pt')

            model.train()



# def train_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     model = model.cuda()
#
#     best_val_loss = float('inf')
#     best_model_state = None
#     eval_every = 50
#
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         counter = 0
#
#         for batch in train_loader:
#             total_start_time = time.time()
#             input_ids = batch['input_ids'].cuda()
#             attention_mask = batch['attention_mask'].cuda()
#             targets = batch['target'].cuda()
#
#             optimizer.zero_grad()
#             start_time = time.time()
#             outputs = model(input_ids, attention_mask)
#             print(f'Forward pass time: {time.time() - start_time:.4f}')
#             mse = torch.mean((outputs.squeeze() - targets) ** 2)
#
#             # Calculate Root Mean Squared Error
#             loss = torch.sqrt(mse)
#
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             print(f'Batch time: {time.time() - total_start_time:.4f}')
#
#             counter += 1
#
#             if counter % eval_every == 0:
#
#                 # Validation
#                 model.eval()
#                 val_loss = 0
#                 with torch.no_grad():
#                     for batch in val_loader:
#                         input_ids = batch['input_ids'].cuda()
#                         attention_mask = batch['attention_mask'].cuda()
#                         targets = batch['target'].cuda()
#
#                         outputs = model(input_ids, attention_mask)
#
#                         mse = torch.mean((outputs.squeeze() - targets) ** 2)
#                         # Calculate Root Mean Squared Error
#                         loss = torch.sqrt(mse)
#
#                         val_loss += loss.item()
#
#                 avg_train_loss = total_loss / eval_every
#                 avg_val_loss = val_loss / len(val_loader)
#
#                 total_loss = 0
#
#                 print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
#
#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     best_model_state = model.state_dict().copy()
#
#                 model.train()
#
#     # Load best model
#     model.load_state_dict(best_model_state)
#     return model


# Example usage:
def main():
    # Assuming you have your data in these formats:
    # texts = ["your text 1", "your text 2", ...]  # List of input texts
    # targets = [0.5, -0.3, ...]  # List of target values between -1 and 1

    # Initialize tokenizer and model

    model_name = 'allenai/longformer-base-4096'
    model = TextRegressionModel(model_name)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    train = pl.read_csv('../../um-game-playing-strength-of-mcts-variants/train.csv')
    print('Shape before dropping columns:', train.shape)

    train = train.select(["Id", "utility_agent1", "EnglishRules"]).to_pandas()
    train = train.reset_index(drop=True)
    test_ids = get_test_indexes_from_file('../data_splits/test_set.txt')

    fold_path = f'../data_splits/Fold_0.txt'
    fold_ids = get_test_indexes_from_file(fold_path)

    general_train_fold = train[(~train['Id'].isin(test_ids))]
    val_fold = general_train_fold.loc[train['Id'].isin(fold_ids)].copy()
    val_fold = val_fold.reset_index(drop=True)
    train_fold = general_train_fold.loc[(~train['Id'].isin(fold_ids))].copy()
    train_fold = train_fold.reset_index(drop=True)

    print('Shape of train fold:', train_fold.shape)
    print('Shape of val fold:', val_fold.shape)

    # Prepare the data
    tokenizer, train_data_loader = prepare_data(train_fold['EnglishRules'].tolist(), train_fold['utility_agent1'].tolist())
    tokenizer, val_data_loader = prepare_data(val_fold['EnglishRules'].tolist(), val_fold['utility_agent1'].tolist())

    # Train model

    for epoch in range(20):
        train_epoch_fast(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()