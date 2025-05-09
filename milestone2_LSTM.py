import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


df = pd.read_csv('/Users/jackiehe/Desktop/Materials/DSUA 301 Advanced Techniques in ML/final project/goldprice2001.2.1--2024.1.1.csv', parse_dates=['Date'], index_col='Date')
data = df.values.reshape(-1, df.shape[1])
# Assume data is a 2D NumPy array
first_col = data[:, 0].reshape(-1, 1)      # gold price or target
other_cols = data[:, 1:]                   # the rest of the features

# print (df,data)

scaler_target = MinMaxScaler()
scaled_first_col = scaler_target.fit_transform(first_col)
scaler_features = MinMaxScaler()
scaled_other_cols = scaler_features.fit_transform(other_cols)
combined_scaled =np.hstack([scaled_first_col, scaled_other_cols])

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_length],
                self.data[idx+self.seq_length][0])

batchsizelist=[32,64,128,256]
learningratelist=[0.1,0.01,0.001,0.0001]
rmsel=[]
mael=[]
r2l=[]
parasl=[]
# finetune on batch size and learning rate
for b in batchsizelist:
    for l in learningratelist:
        SEQ_LENGTH = 10
        BATCH_SIZE = b
        EPOCHS = 300
        LR = l
        print("batch size is",BATCH_SIZE,"learning rate is",LR)
        dataset = TimeSeriesDataset(combined_scaled, SEQ_LENGTH)
        train_size = int(len(dataset) * 0.7)
        train_set = torch.utils.data.Subset(dataset, list(range(train_size)))
        val_size=int(len(dataset)*0.85)
        val_set=torch.utils.data.Subset(dataset, list(range(train_size,val_size)))
        test_set = torch.utils.data.Subset(dataset, list(range(val_size, len(dataset))))


        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,shuffle=False)
        val_loader=DataLoader(val_set, batch_size=1)
        test_loader = DataLoader(test_set, batch_size=1)

        class LSTMModel(nn.Module):
            def __init__(self, input_size=5, hidden_size=50, output_size=1):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.linear = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.linear(out[:, -1])
                return out
            
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
        tl = []
        vl = []
        # for X_batch, y_batch in train_loader:
        #     print("X_batch shape:", X_batch.shape)
        #     print("y_batch shape:", y_batch.shape)
        #     print("X_batch example:", X_batch[0])  # Show the first sample
        #     print("y_batch example:", y_batch[0])  # Show the first label
        #     break  # Only test the first batch


        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.view(-1, SEQ_LENGTH, 5)
                y_batch = y_batch.view(-1, 1)

                output = model(X_batch)
                loss = criterion(output, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_losses.append(loss.item())
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.view(-1, SEQ_LENGTH, 5)
                    y_val = y_val.view(-1, 1)
                    val_output = model(X_val)
                    val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            tl.append(avg_train_loss)
            vl.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.16f} | Val Loss: {avg_val_loss:.16f}")

        model.eval()
        preds, actuals = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.view(-1, SEQ_LENGTH, 5)
                pred = model(X_batch)
                preds.append(pred.item())
                actuals.append(y_batch.item())

        # Inverse scale
        preds = scaler_target.inverse_transform(np.array(preds).reshape(-1, 1))
        actuals = scaler_target.inverse_transform(np.array(actuals).reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds)
        # Plot results
        # plt.plot(tl, label='train loss')
        # plt.plot(vl, label=f'validation loss')
        # plt.legend()
        # plt.title('loss over epoch')
        # plt.show()
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        rmsel.append(rmse)
        mael.append(mae)
        r2l.append(r2)
        parasl.append((b,l))

min=min(rmsel)
min_index=rmsel.index(min)
print(parasl[min_index],min)

SEQ_LENGTH = 10
BATCH_SIZE = 128 #by the best fine tune result
EPOCHS = 600 # give more epoch to train, this value is also finetuned
LR = 0.001
print("batch size is",BATCH_SIZE,"learning rate is",LR)
dataset = TimeSeriesDataset(combined_scaled, SEQ_LENGTH)
train_size = int(len(dataset) * 0.7)
train_set = torch.utils.data.Subset(dataset, list(range(train_size)))
val_size=int(len(dataset)*0.85)
val_set=torch.utils.data.Subset(dataset, list(range(train_size,val_size)))
test_set = torch.utils.data.Subset(dataset, list(range(val_size, len(dataset))))


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,shuffle=False)
val_loader=DataLoader(val_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1])
        return out
    
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
tl = []
vl = []
# for X_batch, y_batch in train_loader:
#     print("X_batch shape:", X_batch.shape)
#     print("y_batch shape:", y_batch.shape)
#     print("X_batch example:", X_batch[0])  # Show the first sample
#     print("y_batch example:", y_batch[0])  # Show the first label
#     break  # Only test the first batch


for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.view(-1, SEQ_LENGTH, 5)
        y_batch = y_batch.view(-1, 1)

        output = model(X_batch)
        loss = criterion(output, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.view(-1, SEQ_LENGTH, 5)
            y_val = y_val.view(-1, 1)
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
    val_losses.append(val_loss.item())
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    tl.append(avg_train_loss)
    vl.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.16f} | Val Loss: {avg_val_loss:.16f}")

model.eval()
preds, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.view(-1, SEQ_LENGTH, 5)
        pred = model(X_batch)
        preds.append(pred.item())
        actuals.append(y_batch.item())

# Inverse scale
preds = scaler_target.inverse_transform(np.array(preds).reshape(-1, 1))
actuals = scaler_target.inverse_transform(np.array(actuals).reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(actuals, preds))
mae = mean_absolute_error(actuals, preds)
r2 = r2_score(actuals, preds)
# Plot results
# plt.plot(tl, label='train loss')
# plt.plot(vl, label=f'validation loss')
# plt.legend()
# plt.title('loss over epoch')
# plt.show()
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
plt.plot(actuals, label='Actual')
plt.plot(preds, label=f'Predicted (RMSE: {rmse:.2f})')
plt.legend()
plt.title('LSTM')
plt.show()

