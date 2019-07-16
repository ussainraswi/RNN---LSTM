# - Another variant of the RNN model: the long short-term memory (LSTM) model.
# - For example, in a time series where the current stock price is
# decided by the historical stock price, where the dependency can be short or long.

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# - The RNN models have hyperparameters, such as the number of
# iterations (EPOCH); batch size dependent on the memory available in a
# single machine; a time step to remember the sequence of information;
# input size, which shows the vector size; and learning rate.

EPOCH = 1
BATCH_SIZE = 64
INPUT_SIZE=28 # rnn input size/ image width
TIME_STEP=28 # rnn time step/ image height
LR=0.01
DOWNLOAD_MNIST=True

# MNIST dataset used as number dataset it has images from 0 to 1.
train_data = dsets.MNIST(root='../../MNIST/MNIST', download=True, train=True, transform=transforms.ToTensor())
# DataLoader: Used to convert train_data as mini-batches, it will reduce the computation time.
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64, num_workers=0)

test_data = dsets.MNIST(root='../../MNIST/MNIST', download=False, train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=64, num_workers=0)

# Volatile:---
# 		-> Variable(test_data.test_data, volatile=True)
# Is recommended for purely inference mode, when you’re sure you won’t be even calling .backward(). 
# It’s more efficient than any other autograd setting - it will use the absolute minimal
# amount of memory to evaluate the model. volatile also determines that requires_grad is False.

test_data_2000 = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255 # size-> [2000 28 28]
test_data_2000_labels = test_data.test_labels.numpy().squeeze()[:2000] # len->2000

# print(train_data.train_data.size()) # torch.Size([60000, 28, 28])
# print(train_data.train_labels.size()) # torch.Size([60000])
# print(train_data.train_labels[0])
# print(train_data.train_data[0].size())
# plt.imshow(train_data.train_data[0], cmap='gray')
# plt.show()

# 60000/64=938
# print(len(train_loader)) # 938
# images, labels = iter(train_loader).next()
# print(labels.size()) # 64 -> Batch Size


# print(len(test_data.test_data[:2000])) # 938
# images, labels = iter(test_loader).next()
# print(labels.size()) # 64 -> Batch Size


class RNN(nn.Module):
    """docstring for RNN."""
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE, # image width
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time step, input size)
        # r_out shape (batch, time step, input size)
        # h_n shape (n_layers, batch, hidden_size) Hidden neurons
        # h_c shape (n_layers, batch, hidden_size) Hidden class
        r_out, (h_n, h_c) = self.rnn(x, None) # None reperesents 0 initial hidden state
        # chosse r_ot at last hidden state
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
# print(RNN())

#     LSTM network, which is proven effective for holding memory for a long time, and
# thus helps in learning.
#   The nn.RNN() model, it hardly learns the parameters, because the vanilla implementation of
# RNN cannot hold or remember the information for a long period of time


# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()


# # Training and Testing
for epoch in range(EPOCH):
    for step, (images, labels) in enumerate(train_loader,0):
        # print('images 1', images.shape) # torch.Size([64, 1, 28, 28])
        images = Variable(images.view(-1, 28 ,28)) # torch.Size([64, 28, 28])
        labels = Variable(labels)

        output = rnn(images)
        loss = loss_func(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50==0:
            test_output = rnn(test_data_2000)
            pred_img, pred_lables = torch.max(test_output, 1)
            accuracy = (pred_lables.numpy().squeeze() == test_data_2000_labels).sum() / float(test_data_2000_labels.size)
            print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, accuracy))

# # SAVE
torch.save(rnn, 'rnn_lstm_full.pkl')
torch.save(rnn.state_dict(), 'rnn_lstm_param.pkl')


# # LOAD
# rnn.load_state_dict(torch.load('rnn_lstm_param.pkl'))


# # TEST
# output = rnn(test_data_2000[:10])
# print('Testing_lables', test_data_2000_labels[:10])

# _, pred_lables = torch.max(output, 1)
# print('Predicted_lables ', pred_lables)