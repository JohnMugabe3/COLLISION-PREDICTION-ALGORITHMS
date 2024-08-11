import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


base_path = '/content/drive/MyDrive/UAV_DELIVERY_DATASET_FULL'
folders = ['UAV_DELIVERY_DATASET_1', 'UAV_DELIVERY_DATASET_2', 'UAV_DELIVERY_DATASET_3']


def process_log_files(base_path, folders):
    headers = ["simt", "id", "type", "lat", "lon", "alt", "tas", "cas", "vs", "gs", "distflown",
               "Temp", "trk", "hdg", "p", "rho", "thrust", "drag", "phase", "fuelflow"]
    all_data = []
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.log'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    df = pd.read_csv(file_path, delimiter=',', names=headers, skiprows=1, comment='#')
                    all_data.append(df)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data


data_path = '/content/drive/MyDrive/UAV_DELIVERY_DATASET_FULL/processed_data.csv'
if not os.path.exists(data_path):
    data = process_log_files(base_path, folders)
    data.to_csv(data_path, index=False)
else:
    data = pd.read_csv(data_path)


def extract_relevant_columns(df):
    if df is not None:
        return df[['lat', 'lon', 'alt']].dropna()
    else:
        return None

data = extract_relevant_columns(data)


def normalize_data(df):
    if df is not None:
        try:
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df['alt'] = pd.to_numeric(df['alt'], errors='coerce')
            df = df.dropna(subset=['lat', 'lon', 'alt']).copy()
            scaler = MinMaxScaler()
            df[['lat', 'lon', 'alt']] = scaler.fit_transform(df[['lat', 'lon', 'alt']])
            return df
        except Exception as e:
            print("Failed to normalize data:", e)
            return None
    else:
        return None

data = normalize_data(data)


normalized_data_path = '/content/drive/MyDrive/UAV_DELIVERY_DATASET_FULL/normalized_data.csv'
if data is not None:
    data.to_csv(normalized_data_path, index=False)
else:
    data = pd.read_csv(normalized_data_path)


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
data_values = data[['lat', 'lon', 'alt']].values
X, y = create_sequences(data_values, seq_length)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


batch_size = 64 
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


input_size = 3
hidden_size = 25  
output_size = 3
num_layers = 1  
num_epochs = 50
learning_rate = 0.001

lstm_model = LSTMPredictor(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)


train_losses = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    lstm_model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = lstm_model(X_batch)
            loss = criterion(output, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        torch.cuda.empty_cache()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()


model_path = '/content/drive/MyDrive/UAV_DELIVERY_DATASET_FULL/lstm_model.pth'
torch.save(lstm_model.state_dict(), model_path)
print(f'Model saved to {model_path}')


lstm_model.eval()
all_predictions = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = lstm_model(X_batch)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
mae = mean_absolute_error(all_targets, all_predictions)
rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')


plt.figure(figsize=(10, 5))
plt.plot(all_targets[:100, 0], label='Actual')
plt.plot(all_predictions[:100, 0], label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()
