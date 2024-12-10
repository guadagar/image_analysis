import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

'''
DL-MBL Image Analyisis course 2024
Excercise: Generate spiral data & use deep net to classify the data
This model is better
Increasing the num of hidden layers (1->2) and the number of nodes(12->64), the performance increase from %50 to 99%
Run in ves_det
w SGD & MSELoss => ~99.4
AdamW &  BCELoss => %99
'''

#Generate spiral data
def generate_spiral_data(n_points, noise=1.0):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points), np.ones(n_points))),
    )

#Generate the plots
def plot_points(Xs, ys, titles):
    num_subplots = len(Xs)
    plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
    for i, (X, y, title) in enumerate(zip(Xs, ys, titles)):
        plt.subplot(1, num_subplots, i + 1)
        plt.title(title)
        plt.plot(X[y == 0, 0], X[y == 0, 1], ".", label="Class 1")
        plt.plot(X[y == 1, 0], X[y == 1, 1], ".", label="Class 2")
        plt.legend()
    plt.show()


X_train, y_train = generate_spiral_data(100)
X_test, y_test = generate_spiral_data(1000)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("GPU not available. Will use CPU.")
    device = torch.device("cpu")


class GoodModel(nn.Module):
    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=2, out_features=64, bias=True), # this layer receives a tensor of size (B, 2) and returns a tensor of size (B, 12)
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64, bias=True), # this layer receives a tensor of size (B, 2) and returns a tensor of size (B, 12)
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1), # this layer receives a tensor of size (B, 12) and returns a tensor of size (B, 1)
            nn.Sigmoid(), # Sigmoid is a non-linear activation function that squashes the output to the range [0, 1], widely used for binary classification
        )
        # Note: the output of the block is a number between 0 and 1. In simplifying terms, you can think of it as "the probability of the input data belonging to class 1".

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

def batch_generator(X, y, batch_size, shuffle=True):
    if shuffle:
        # Shuffle the data at each epoch
        indices = np.random.permutation(len(X))
    else:
        # Process the data in the order as it is
        indices = np.arange(len(X))
    for i in range(0, len(X), batch_size):
        yield X[indices[i : i + batch_size]], y[indices[i : i + batch_size]]

def run_epoch(model, optimizer, X_train, y_train, batch_size, loss_fn, device):
    n_samples = len(X_train)
    total_loss = 0
    # Set the model to training mode, essential when using certain layers
    model.train()

    for X_b, y_b in batch_generator(X_train, y_train, batch_size):
        # Convert the data to PyTorch tensors
        X_b = torch.tensor(X_b, dtype=torch.float32, device=device)
        y_b = torch.tensor(y_b, dtype=torch.float32, device=device)

        # Reset the optimizer state
        optimizer.zero_grad()

        # Forward pass: pass the data through the model and retrieve the prediction
        y_pred = model(X_b).squeeze()

        # Compute the loss function with the prediction and the ground truth
        loss = loss_fn(y_pred, y_b)

        # Backward pass: compute the gradient of the loss w.r.t. the parameters
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the loss (for monitoring purposes)
        total_loss += loss.item() # the .item() converts the single-number Tensor to a Python floating point number, avoiding retaining the computational graph in the loss tensor
    return total_loss / n_samples

# Initialize the model, optimizer and set the loss function
good_model = GoodModel()
# The .to() method will move the model to the appropiate device (e.g. the GPU if available)
good_model.to(device)

#defino el optimizer que uso
optimizer = torch.optim.SGD(
    good_model.parameters(), lr=0.01
)  # SGD - Stochastic Gradient Descent

#defino el loss function que uso
loss_fn = nn.MSELoss(reduction="sum")  # MSELoss - Mean Squared Error Loss

#optimizer = torch.optim.AdamW(good_model.parameters(), lr=0.001)
#loss_fn = nn.BCELoss(reduction="sum")  # Binary Cross Entropy Loss


batch_size = 10
num_epochs = 1500

for epoch in (pbar := tqdm(range(num_epochs), total=num_epochs, desc="Training")):
    # Run an epoch over the training set
    curr_loss = run_epoch(
        good_model, optimizer, X_train, y_train, batch_size, loss_fn, device
    )

    # Update the progress bar to display the training loss
    pbar.set_postfix({"training loss": curr_loss})

#how to evaluate the performance?
def predict(model, X, y, batch_size, device):
    predictions = np.empty((0,))
    model.eval() # set the model to evaluation mode
    with torch.inference_mode(): # this "context manager" is used to disable gradient computation (among others), which is not needed during inference and offers improved performance
        for X_b, y_b in batch_generator(X, y, batch_size, shuffle=False):
            X_b = torch.tensor(X_b, dtype=torch.float32, device=device)
            y_b = torch.tensor(y_b, dtype=torch.float32, device=device)
            y_pred = model(X_b).squeeze().detach().cpu().numpy()
            predictions = np.concatenate((predictions, y_pred), axis=0)
    return np.round(predictions)

def accuracy(y_pred, y_gt):
    return np.sum(y_pred == y_gt) / len(y_gt)

good_predictions = predict(good_model, X_test, y_test, batch_size, device)
good_accuracy = accuracy(good_predictions, y_test)

plot_points(
    [X_test, X_test],
    [y_test, good_predictions],
    ["Testing data", f"Good Model Classification ({good_accuracy * 100:.2f}% correct)"],
)

plt.show()
