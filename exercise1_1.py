#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import load_mnist
import math
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

if  __name__ == '__main__':
    print("Hello world")
    X_train, Y_train, X_test, Y_test = load_mnist.load_mnist()
    Y_train = Y_train.argmax(axis=1)
    Y_test = Y_test.argmax(axis=1)
    X_train = torch.from_numpy(X_train.astype(float))
    Y_train = torch.from_numpy(Y_train)
    X_test = torch.from_numpy(X_test.astype(float))
    Y_test = torch.from_numpy(Y_test)
    lr=0.01
    batch_size=1000
    iterations=1000
    random_index = np.arange(X_train.shape[0])
    np.random.shuffle(random_index)
    EPOCHS = math.floor(batch_size*iterations/X_train.shape[0])
    ITERATION_PER_EPOCH = math.floor(X_train.shape[0]/batch_size)
    net = Net()
    # net.parameters is all the parameters that can be changed
    optimizer = optim.Adam(net.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    it = EPOCHS*ITERATION_PER_EPOCH
    loss_train=np.zeros(it)
    acc_train=np.zeros(it)
    loss_test=np.zeros(it)
    acc_test=np.zeros(it)

    for epoch in range(EPOCHS):
        np.random.shuffle(random_index)
        for iteration in range(ITERATION_PER_EPOCH):
            X_batch = X_train[random_index[iteration*batch_size:(iteration+1)*batch_size],:]
            Y_batch = Y_train[random_index[iteration*batch_size:(iteration+1)*batch_size]]
            net.zero_grad()
            output = net(X_batch.float())
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            loss_train[epoch*ITERATION_PER_EPOCH+iteration] = loss.item()
            output = output.argmax(axis=1)
            correct = torch.sum(Y_batch==output)
            acc_train[epoch*ITERATION_PER_EPOCH+iteration] = correct/batch_size

            output_test = net(X_test.float())
            loss_test_it = criterion(output_test, Y_test)
            loss_test[epoch*ITERATION_PER_EPOCH+iteration] = loss_test_it.item()
            output_test = output_test.argmax(axis=1)
            correct_test = torch.sum(Y_test==output_test)
            acc_test[epoch*ITERATION_PER_EPOCH+iteration] = correct_test/10000

    X_plt = np.arange(it)
    plt.plot(X_plt, acc_train, label="Train")
    plt.plot(X_plt, acc_test, label="Test")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over iterations")
    plt.legend()
    plt.show()

    plt.plot(X_plt, loss_train, label="Train")
    plt.plot(X_plt, loss_test, label="Test")
    plt.xlabel("Cost")
    plt.ylabel("Accuracy")
    plt.title("Cost over iterations")
    plt.legend()
    plt.show()
    #Test acc: 0.902
    #6min 50s

    #Acc: 0.97
    #1min 9s
    print(f"Accuracy: {acc_test[-1]}")
    print(f"Cost: {loss_test[-1]}")
