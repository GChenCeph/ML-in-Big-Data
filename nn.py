import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Neural Network Model
class RegressionNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    features = df.drop(columns=[f'f{i}' for i in range(1, 13)]).values
    targets = df[[f'f{i}' for i in range(1, 13)]].values
    return features, targets

# Prepare Data
scaler = StandardScaler()
X_train, y_train = load_data('training.csv')
X_train = scaler.fit_transform(X_train)
X_test, y_test = load_data('testing.csv')
X_test = scaler.transform(X_test)
y_reference = pd.read_csv('reference.csv')[[f'f{i}' for i in range(1, 13)]].values

train_dataset = CustomDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = CustomDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model
input_size = X_train.shape[1]
output_size = y_train.shape[1]
model = RegressionNN(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0055)

# Train Model
num_epochs = 100
for epoch in range(num_epochs):
    for features, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate Model
model.eval()
total_loss = 0
predictions = []
with torch.no_grad():
    for features, targets in test_loader:
        outputs = model(features)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        # Round the outputs to the nearest integer
        rounded_outputs = torch.round(outputs).numpy()
        predictions.extend(rounded_outputs)

avg_loss = total_loss / len(test_loader)
print(f'Average Testing Loss: {avg_loss}')

# Compare with Reference and Calculate Accuracy
accuracy_list = []
for i in range(12):
    correct = sum(1 for j in range(len(predictions)) if predictions[j][i] == y_reference[j][i])
    accuracy = correct / len(predictions)
    accuracy_list.append(accuracy)

# Save results
with open('nn_result.txt', 'w') as file:
    file.write(f'Average Testing Loss: {avg_loss}\n')
    file.write('Monthly Accuracy:\n')
    for i, acc in enumerate(accuracy_list, 1):
        file.write(f'Month {i}: {acc:.2f}\n')

print("Results written to nn_result.txt")
