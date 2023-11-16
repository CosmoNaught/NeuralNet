import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set this flag to True to load the model from cache if available, else False
load_from_cache = False

csv_file = 'D:/NeuralNet/SIR_model/R/archive/model/20231115-203321-f7d9d2df/outputs/epidemic_curves.csv'
data = pd.DataFrame(pd.read_csv(csv_file))

# Preprocess the Data
features = data[['beta', 'gamma']].values
targets = data.drop(['beta', 'gamma'], axis=1).values

scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()
features_scaled = scaler_features.fit_transform(features)
targets_scaled = scaler_targets.fit_transform(targets)

features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)

indices = np.arange(len(data))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(features_tensor, targets_tensor, indices, test_size=0.5, random_state=42)

# Dataset and DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_data_with_indices = TensorDataset(X_test, y_test, torch.arange(len(X_test)))
test_loader = DataLoader(test_data_with_indices, shuffle=True, batch_size=batch_size)


# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob = 0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x, _ = self.lstm(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc(x.view(len(x), -1))
        x = self.softplus(x)
        return x

# Function to save the model
def save_model(model, filename="ML_model/Python/cache/model_LSTM.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)

# Function to load the model
def load_model(model, filename="ML_model/Python/cache/model_LSTM.pth"):
    model.load_state_dict(torch.load(filename))
    return model

# Check if model is available in cache and load_from_cache flag is set
model_path = "ML_model/Python/cache/model_LSTM.pth"
model_exists = os.path.isfile(model_path) and load_from_cache

# Initialize Model, Loss, Optimizer
input_size = 2
hidden_layer_size = 50
output_size = targets.shape[1]  # Number of days

model = LSTM(input_size, hidden_layer_size, output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if model_exists:
    print("Loading pre-trained model")
    model = load_model(model, model_path)
else:
    print("No model found in cache or load_from_cache is set to False. Training model.")
    # Train the Model
    epochs = 32

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            y_pred = model(inputs)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            
        if epoch % 1 == 0:  # Modify as per requirement for less verbose output
            print(f'Epoch {epoch+1} loss: {single_loss.item()}')

    # Save the trained model
    save_model(model, model_path)

# Run Emulator Function
def run_emulator(model, test_loader, device, scaler_targets):
    model.eval()
    predictions = []
    actual = []
    settings = []  # This will store the beta and gamma values

    with torch.no_grad():
        for inputs, targets, batch_indices in tqdm(test_loader, desc='Running emulator'):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Denormalizing the predictions and actual values
            predictions_denorm = scaler_targets.inverse_transform(outputs.cpu().numpy())
            actual_denorm = scaler_targets.inverse_transform(targets.cpu().numpy())

            predictions.append(predictions_denorm)
            actual.append(actual_denorm)
            settings.append(inputs.cpu().numpy())  # Assuming inputs need not be denormalized

    return predictions, actual, settings

def plot_mint_compare(predictions, actual, settings, num_samples=9):
	
    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Select random row indices
    row_indices = np.random.choice(flattened_predictions.shape[0], size=num_samples, replace=False)

    # Create a 3x3 grid of scatter plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Scatterplots of actual vs predicted values for selected time series')

    for i, row_index in enumerate(row_indices):
        # Get the selected row from the predictions and actual arrays
        selected_predictions = flattened_predictions[row_index]
        selected_actual = flattened_actual[row_index]

        ax = axs[i//3, i%3]
        ax.scatter(selected_actual, selected_predictions, c='black', alpha=0.5, marker='o', label='Predicted')
        ax.scatter(selected_actual, selected_actual, c='red', alpha=0.5, marker='o', label='Actual')
        ax.plot([np.min(selected_actual), np.max(selected_actual)], [np.min(selected_actual), np.max(selected_actual)], c='gray', linestyle='--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Scatterplot #{row_index}')
        ax.legend()

    plt.show()

def plot_mint_time_series(predictions, actual, settings, num_samples=9):
    
    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Select random row indices
    row_indices = np.random.choice(flattened_predictions.shape[0], size=num_samples, replace=False)

    # Create a 3x3 grid of time series plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Time series of actual vs predicted values for selected time series')

    for i, row_index in enumerate(row_indices):
        # Get the selected row from the predictions and actual arrays
        selected_predictions = flattened_predictions[row_index]
        selected_actual = flattened_actual[row_index]

        ax = axs[i//3, i%3]
        ax.plot(selected_predictions, c='black', label='Predictions', alpha=0.5, linestyle='-', marker='o', markersize=4)
        ax.plot(selected_actual, c='red', label='Actual', alpha=0.5, linestyle='-', marker='o', markersize=4)
        ax.set_xlabel('Year')
        ax.set_ylabel('Values')
        ax.set_title(f'Time series #{row_index}')

        # Adjust the x-axis tick labels
        tick_positions = np.arange(0, len(selected_predictions), 12)
        tick_labels = [2017 + i for i in tick_positions // 12]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        ax.legend()

    plt.show()

# Run the emulator and plot
predictions, actual, settings = run_emulator(model, test_loader, device, scaler_targets)
plot_mint_compare(predictions, actual, settings)
plot_mint_time_series(predictions, actual, settings, num_samples=9)

def generate_and_plot_series(beta, gamma, model, scaler_features, scaler_targets, num_days=100):
    # Normalize the input features
    features = np.array([[beta, gamma]])
    features_scaled = scaler_features.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    # Generate prediction
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(features_tensor)
        prediction = scaler_targets.inverse_transform(prediction_scaled.cpu().numpy())

    # Plot the time-series
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_days), prediction[0], label='Predicted Time-Series', color='blue')
    plt.xlabel('Day')
    plt.ylabel('Values')
    plt.title(f'Time-Series for Beta: {beta}, Gamma: {gamma}')
    plt.legend()
    plt.show()

beta_value = 0.2  # Example beta value
gamma_value = 0.1  # Example gamma value
generate_and_plot_series(beta_value, gamma_value, model, scaler_features, scaler_targets)
