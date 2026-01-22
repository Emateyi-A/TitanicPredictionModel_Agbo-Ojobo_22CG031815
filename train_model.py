import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

data_1 = load_iris()
X = data_1.data[:, :3] 
y = (data_1.target == 0).astype(int) 


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)


class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


model = TitanicModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 100
for epoch in range(epochs):
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    train_outputs = model(X_train)
    train_accuracy = ((train_outputs > 0.5) == y_train).float().mean()
    
    test_outputs = model(X_test)
    test_accuracy = ((test_outputs > 0.5) == y_test).float().mean()
    
    print(f"\nThe Training Accuracy: {train_accuracy:.4f}")
    print(f"The Testing Accuracy: {test_accuracy:.4f}")


torch.save(model.state_dict(), "model.pth")
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nThe Model saved as 'model.pth'")
print("The Scaler saved as 'scaler.pkl'")
