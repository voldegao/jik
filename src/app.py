import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import requests
import numpy as np
import time
import os
import sys

# Fetch data
def fetch_data(api_url):
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

# Preprocess data
def preprocess_data(data):
    features = torch.tensor(data['x'], dtype=torch.float32)
    labels = torch.tensor(data['y'], dtype=torch.float32)
    return features, labels

# Create PyTorch Dataset
def create_torch_dataset(features, labels, batch_size=1024):
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Rescale predictions (placeholder for any required post-processing)
def rescale_predictions(predictions):
    return predictions

# Predict function
def predict(model, test_data, device):
    model.eval()
    with torch.no_grad():
        test_data = test_data.to(device)
        predictions = model(test_data)
    return rescale_predictions(predictions)

# Define PyTorch model
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# Main Script
def run(api_url):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Training Start")
    raw_data = fetch_data(api_url)
    print("Data Fetched")
    features, labels = preprocess_data(raw_data)
    dataloader = create_torch_dataset(features, labels)

    model = SimpleModel(input_size=features.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Train the model
    start_time = time.time()
    model.train()
    epochs = 100

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (batch_features, batch_labels) in enumerate(dataloader):
            # Move data to GPU
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_features).squeeze()
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Update the progress for the epoch
        avg_loss = epoch_loss / len(dataloader)
        sys.stdout.write(f"\rEpoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        sys.stdout.flush()

    print("\nTraining Completed")
    end_time = time.time()
    print(f"Model Trained in {end_time - start_time:.2f} seconds")

    # Save model (cloud storage or skip saving if unnecessary)
    save_path = os.path.join(os.getcwd(), "trained_model1.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at: {save_path}")

    # Test prediction
    test_data = torch.tensor([[2, 12, 1, 1, 1, 8]], dtype=torch.float32).to(device)  # Replace with actual test input
    prediction = predict(model, test_data, device)
    print("Prediction:", prediction.item())

if __name__ == "__main__":
    api_url = os.getenv("API_URL", "https://rnqgo-154-144-224-93.a.free.pinggy.link/api/visilog/ml/data/attente_externe")
    run(api_url)
