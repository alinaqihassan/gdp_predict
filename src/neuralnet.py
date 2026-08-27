# neuralnet.py
# implements data processing and the main training and testing loops for the model
# provides a MASE test value and plots the predictions of the model vs the naïve model
# and the true value, as well as comparing the error of the model to that of the naïve model

import torch, json
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# hyperparameters
learning_rate = 1e-3
batch_size = 16
epochs = 20000

# setting up a dataset class to hold data and be queried by the model
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, prev_vals):
        self.X = X
        self.y = y
        self.prev_vals = prev_vals
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.prev_vals[idx]

# Defining neural network structure and forward propagation
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_lrelu_stack = nn.Sequential(
            nn.Linear(3*4, 6),
            nn.LeakyReLU(0.05),
            nn.Linear(6, 6),
            nn.LeakyReLU(0.05),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_lrelu_stack(x)
        return logits

# standardise the data and create a sliding window view then create inputs and outputs arrays
def create_dataset(data, window=4):
    eps = 1e-8

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + eps
    data_standardised = (data - mean) / std

    z = torch.from_numpy(data_standardised).float().unfold(0, window + 1, 1)

    X = z[:, :, :window].permute(0, 2, 1)

    prev_vals_std = data_standardised[window-1:-1, 0]
    next_vals_std = data_standardised[window:, 0]

    y = next_vals_std - prev_vals_std

    return X, torch.tensor(y).float(), torch.tensor(prev_vals_std).float(), mean, std

# train the model and print training loss data
def train_loop(dataloader, model, loss_fn, optimiser, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device=device)
        y = y.to(device=device).unsqueeze(1)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# test the model after each epoch and print testing loss
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y, _ in dataloader:
            X = X.to(device=device)
            y = y.to(device=device).unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

# test the model after training and calculate and print MASE value
def mase(dataloader, model, device, mean, std):
    model.eval()
    
    abs_errors = []
    naive_errors = []

    true_vals = []
    naive_vals = []
    predicted_vals = []

    with torch.no_grad():
        for X, y, prev_vals_batch in dataloader:
            X = X.to(device=device)
            y = y.to(device=device).unsqueeze(1)

            pred = model(X)

            y_np = y.cpu().numpy().reshape(-1, 1)
            pred_np = pred.cpu().numpy().reshape(-1, 1)
            prev_np = prev_vals_batch.cpu().numpy().reshape(-1, 1)

            prev_actual = prev_np * std[0] + mean[0]
            pred_actual = (prev_np + pred_np) * std[0] + mean[0]
            true_actual = (prev_np + y_np) * std[0] + mean[0]
            naive_actual = prev_actual
            
            abs_errors.append(np.abs(pred_actual - true_actual))
            naive_errors.append(np.abs(naive_actual - true_actual))

            true_vals.append(true_actual)
            predicted_vals.append(pred_actual)
            naive_vals.append(naive_actual)
    
    abs_errors = np.concatenate(abs_errors)
    naive_errors = np.concatenate(naive_errors)

    true_vals = np.concatenate(true_vals).flatten()
    naive_vals = np.concatenate(naive_vals).flatten()
    predicted_vals = np.concatenate(predicted_vals).flatten()

    mae_model = np.mean(abs_errors)
    mae_naive = np.mean(naive_errors)

    mase_value = mae_model / mae_naive

    return mase_value, true_vals, naive_vals, predicted_vals

with open("../data/quarterly_data.json", "r") as file:
    dict_data = json.load(file)

data_list = list(dict_data.values()) 

data = np.array(data_list)

X, y, prev_vals, mean, std = create_dataset(data)

split = int(0.8*len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
prev_train, prev_test = prev_vals[:split], prev_vals[split:]

training_dataset = TimeSeriesDataset(X_train, y_train, prev_train)
test_dataset = TimeSeriesDataset(X_test, y_test, prev_test)

train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu" # use Nvidia GPU if available
print(f"Using {device} device")

model = NeuralNetwork().to(device) # move the model to the GPU if available

loss_fn = nn.MSELoss() # mean squared error loss function

optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Descent

for t in range(epochs): # loop through each epoch, training the model and testing (validating)
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimiser, device)
    test_loop(test_dataloader, model, loss_fn, device)
print("Done!")

mase, true_vals, naive_vals, predicted_vals = mase(test_dataloader, model, device, mean, std)

print(f"MASE: {mase:6f}")

dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in list(dict_data.keys())[4 + split:]]

naive_error = list(naive_vals - true_vals)
model_error = list(predicted_vals - true_vals)

true_vals = list(true_vals)
naive_vals = list(naive_vals)
predicted_vals = list(predicted_vals)

# plot the naïve error compared to the model's error and the naïve and model predictions compared to the true values
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator(base=1))

ax1.plot(dates, true_vals)
ax1.plot(dates, naive_vals)
ax1.plot(dates, predicted_vals)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(base=1))

ax2.plot(dates, naive_error)
ax2.plot(dates, model_error)

fig.autofmt_xdate()

plt.show()