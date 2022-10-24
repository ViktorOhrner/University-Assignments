#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from sklearn.metrics import confusion_matrix

import load_mnist

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.fc = nn.Linear(1568, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, stride=(2, 2))
        x = F.relu(self.conv3(x))
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_mnist.load_mnist()
    Y_train = Y_train.argmax(axis=1)
    Y_test = Y_test.argmax(axis=1)
    X_train = torch.from_numpy(X_train.astype(float))
    Y_train = torch.from_numpy(Y_train)
    X_test = torch.from_numpy(X_test.astype(float))
    Y_test = torch.from_numpy(Y_test)
    random_index = torch.randperm(X_train.size(0))

    lr = 0.01
    batch_size = 80
    iterations = 1500
    print(X_train.shape[0])
    EPOCHS = math.floor(batch_size*iterations/X_train.shape[0])
    ITERATION_PER_EPOCH = math.floor(X_train.shape[0]/batch_size)

    x = torch.rand(1, 1, 28,28)
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    it = EPOCHS*ITERATION_PER_EPOCH
    loss_train=torch.zeros(it)
    acc_train=torch.zeros(it)
    loss_test=torch.zeros(it)
    acc_test=torch.zeros(it)

    print(f"Epochs: {EPOCHS}")
    print(f"Iterations per epoch: {ITERATION_PER_EPOCH}")


    time_start=process_time()
    for epoch in range(EPOCHS):
        random_index = torch.randperm(X_train.size(0))
        for iteration in range(ITERATION_PER_EPOCH):
            print(f"Iteration: {epoch*ITERATION_PER_EPOCH+iteration}")
            X_batch = X_train[random_index[iteration*batch_size:(iteration+1)*batch_size],:]
            Y_batch = Y_train[random_index[iteration*batch_size:(iteration+1)*batch_size]]
            net.zero_grad()
            # print(X_batch.shape)
            output = net(X_batch.view(batch_size, 1, 28, 28).float())
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            loss_train[epoch*ITERATION_PER_EPOCH+iteration] = loss.item()
            output = output.argmax(axis=1)
            correct = torch.sum(Y_batch==output)
            acc_train[epoch*ITERATION_PER_EPOCH+iteration] = correct/batch_size

            output_test = net(X_test.view(X_test.size(0), 1, 28, 28).float())
            loss_test_it = criterion(output_test, Y_test)
            loss_test[epoch*ITERATION_PER_EPOCH+iteration] = loss_test_it.item()
            output_test = output_test.argmax(axis=1)
            correct_test = torch.sum(Y_test==output_test)
            acc_test[epoch*ITERATION_PER_EPOCH+iteration] = correct_test/X_test.size(0)

    time_stop=process_time()
    time_elapsed = time_stop-time_start

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
    plt.ylabel("Cost")
    plt.title("Cost over iterations")
    plt.legend()
    plt.show()
    #Exercise 1.5
    #Test acc: 0.985
    #Test cost: 0.0469
    #Time: 50 min

    print(f"Accuracy: {acc_test[-1]}")
    print(f"Cost: {loss_test[-1]}")
    print(f"Time: {time_elapsed}")


    index_guesses = Y_test==output_test
    correct_index = (index_guesses==True).nonzero()
    wrong_index = (index_guesses==False).nonzero()

    plt.subplot(2, 2, 1)
    img = X_test[wrong_index[0, 0]]
    imgplot = plt.imshow(img.view(28, 28))
    print("Correct guess: ", output_test[wrong_index[0, 0]])
    plt.subplot(2, 2, 2)
    img = X_test[wrong_index[1, 0]]
    imgplot = plt.imshow(img.view(28, 28))
    print("Wrong guess: ", output_test[wrong_index[1, 0]])
    plt.subplot(2, 2, 3)
    img = X_test[wrong_index[2, 0]]
    imgplot = plt.imshow(img.view(28, 28))
    print("Correct guess: ", output_test[wrong_index[2, 0]])
    plt.subplot(2, 2, 4)
    img = X_test[wrong_index[3, 0]]
    imgplot = plt.imshow(img.view(28, 28))
    print("Correct guess: ", output_test[wrong_index[3, 0]])
    plt.show()


    print(confusion_matrix(Y_test, output_test))
