import torch, json
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

learning_rate = 0.005
batch_size = 16

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_lrelu_stack = nn.Sequential(
            nn.Linear(3*4, 9),
            nn.LeakyReLU(0.05),
            nn.Linear(9, 6),
            nn.LeakyReLU(0.05),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_lrelu_stack(x)
        return logits

def normalise(data):
    minim = np.min(data, 0)
    maxim = np.max(data, 0)
    valrange = maxim - minim
    return (data - minim)/valrange, valrange, minim

def create_dataset(data, window=4):
    z = torch.from_numpy(data).to(torch.float32).unfold(0, window + 1, 1)

    X = z[:, :, :window].permute(0, 2, 1)
    y = z[:, 0, window]

    return X, y

def pytorch_rolling_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,step_size)

def train_loop(dataloader, model, loss_fn, optimiser, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalisation and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
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

def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device=device)
            y = y.to(device=device).unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

with open("../data/quarterly_data.json", "r") as file:
    dict_data = json.load(file)

data_list = list(dict_data.values())

raw_data = np.array(data_list)

data, vranges, mins = normalise(raw_data)

X, y = create_dataset(data)

split = int(0.8*data.shape[0])

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

training_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()

optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimiser, device)
    test_loop(test_dataloader, model, loss_fn, device)
print("Done!")

test = np.array([[[14715.058, 0.7112145553268009, 0.7749999999999999], [14706.538, 1.13677364185591, 1.5608196721311474], [14865.701, 2.19534465606688, 2.23140625], [14898.999, 1.16305534624079, 2.341875]]])

model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy((test - mins)/vranges)
                   .to(device=device, dtype=torch.float32))
output = logits.cpu().detach().numpy()
print(output*vranges[0]+mins[0])