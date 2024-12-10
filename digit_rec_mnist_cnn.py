import torch
from torch import nn
import torch.nn.functional as F
import math
import pickle
import gzip
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''Learning pytorch: classifying handwritten images - with class & optim for update parameters
GCG
07.24
'''
#Load the dataset - data:numpy array
with gzip.open('mnist.pkl.gz', "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

#to tensor
features_train_tensor = torch.tensor(x_train)
target_train_tensor = torch.tensor(y_train)
#dataset wrapping tensor in pytorch
train_ds = TensorDataset(features_train_tensor, target_train_tensor)
#dataloader for managing batches in pytorch
bs = 64
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

#to validate
features_test_tensor = torch.tensor(x_valid)
target_test_tensor = torch.tensor(y_valid)
#dataset wrapping tensor in pytorch
valid_ds = TensorDataset(features_test_tensor, target_test_tensor)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


n, c = x_train.shape

#set hyperparameters
lr = 0.5  # learning rate
epochs = 2

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

#lass Mnist_Logistic(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.lin = nn.Linear(784, 10)
#
#    def forward(self, xb):
#        return self.lin(xb)

model = Mnist_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
         # get the inputs
         inputs, labels = data

#         # zero the parameter gradients
         optimizer.zero_grad()
#
#         # forward + backward + optimize
         outputs = model(inputs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()
#
#        # print statistics
         running_loss += loss.item()
         if i % 2000 == 1999:    # print every 2000 mini-batches
             print('[%d, %5d] loss: %.3f' %
                   (epoch + 1, i + 1, running_loss / 2000))
             running_loss = 0.0

print('Finished Training')
#save the trained parameters
torch.save(model.state_dict(),'trained_net_mnist_cnn.pth')

#Evaluation
model = Mnist_CNN()
model.load_state_dict(torch.load('trained_net_mnist_cnn.pth'))
#
correct = 0
total = 0
model.eval()
#
with torch.no_grad():
     for data in valid_dl:
         images, labels = data
         outputs = model(images)
         _,predicted = torch.max(outputs,1)
         total += labels.size(0)
         correct += (predicted == labels).sum().item()
#
accuracy = 100* correct / total
print(f'accuracy: {accuracy}%')
