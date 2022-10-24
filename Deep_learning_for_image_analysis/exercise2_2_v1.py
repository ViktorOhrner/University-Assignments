#!/usr/bin/env python3
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import load_warwick
from torch.utils.data import Dataset
from skimage import io


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.tconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.convFinal = nn.Conv2d(8, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=(2, 2)))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=(2, 2)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        # x = F.relu(self.convFinal(x))
        x  = self.convFinal(x)
        # print("size", x.size())
        # x = F.softmax(x, dim=1)
        # imgplot = plt.imshow(img.detach())
        # plt.show()
        # print(x.size())
        return x

def calc_dice_coef(pred, target):
    num_same = torch.sum(torch.eq(pred, target))
    return num_same * 2 / (2 * torch.numel(pred))

if __name__ == "__main__":
    # LOAD THE DATASET
    X_train, Y_train, X_test, Y_test = load_warwick.load_warwick()

    validation_fraction = 0.2
    random_index = torch.randperm(X_train.size(0))
    print("Random index size: ", random_index[17:].size())
    validation_index = math.floor(X_train.size(0)*validation_fraction)
    X_valid = X_train[random_index[0:validation_index]]
    X_train = X_train[random_index[validation_index:]]
    Y_valid = Y_train[random_index[0:validation_index]]
    Y_train = Y_train[random_index[validation_index:]]

    random_rot = torch.rand(X_train.size(0))
    num_to_rot = sum(torch.where(random_rot>0.25, 1, 0))
    X_temp = torch.zeros(X_train.size(0)+num_to_rot, X_train.size(1), X_train.size(2), X_train.size(3))
    Y_temp = torch.zeros(Y_train.size(0)+num_to_rot, Y_train.size(1), Y_train.size(2))
    for i in range(num_to_rot):
        if random_rot[i] > 0.75:
            X_temp[X_train.size(0)+i] = torch.rot90(X_train[i], 3, (1, 2))
            Y_temp[Y_train.size(0)+i] = torch.rot90(Y_train[i], 3)
        elif random_rot[i] > 0.5:
            X_temp[X_train.size(0)+i] = torch.rot90(X_train[i], 2, (1, 2))
            Y_temp[Y_train.size(0)+i] = torch.rot90(Y_train[i], 2)
        elif random_rot[i] > 0.25:
            X_temp[X_train.size(0)+i] = torch.rot90(X_train[i], 1, (1, 2))
            Y_temp[Y_train.size(0)+i] = torch.rot90(Y_train[i], 1)
    X_temp[:X_train.size(0)] = X_train
    Y_temp[:Y_train.size(0)] = Y_train
    X_train = X_temp.float()
    Y_train = Y_temp.to(torch.long)

    lr = 0.01
    batch_size = 5
    iterations = 5000
    EPOCHS = math.floor(batch_size*iterations/X_train.shape[0])
    ITERATION_PER_EPOCH = math.floor(X_train.shape[0]/batch_size)

    net = Net()
    net = net.float()
    # optimizer = optim.Adam(net.parameters(), lr)
    optimizer = optim.AdamW(net.parameters(), lr, weight_decay=0.001)
    # optimizer = optim.SGD(net.parameters(), lr, momentum=0.9) # Try this one
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    it = EPOCHS*ITERATION_PER_EPOCH
    loss_train=torch.zeros(it)
    acc_train=torch.zeros(it)
    dice_train=torch.zeros(it)
    loss_valid=torch.zeros(it)
    acc_valid=torch.zeros(it)
    dice_valid=torch.zeros(it)

    print(f"Epochs: {EPOCHS}")
    print(f"Iterations per epoch: {ITERATION_PER_EPOCH}")

    for epoch in range(EPOCHS):
        random_index = torch.randperm(X_train.size(0))
        for iteration in range(ITERATION_PER_EPOCH):
            print(f"Iteration: {epoch*ITERATION_PER_EPOCH+iteration}")
            X_batch = X_train[random_index[iteration*batch_size:(iteration+1)*batch_size],:,:,:]
            Y_batch = Y_train[random_index[iteration*batch_size:(iteration+1)*batch_size],:,:]
            net.zero_grad()
            output = net(X_batch.float())
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            loss_train[epoch*ITERATION_PER_EPOCH+iteration] = loss.item()
            output = output.argmax(axis=1)

            correct = torch.sum(Y_batch==output)
            acc_train[epoch*ITERATION_PER_EPOCH+iteration] = correct/batch_size
            dice_train[epoch*ITERATION_PER_EPOCH+iteration] = calc_dice_coef(output, Y_batch)

            output_valid = net(X_valid.float())
            loss_valid_it = criterion(output_valid.squeeze(), Y_valid)
            loss_valid[epoch*ITERATION_PER_EPOCH+iteration] = loss_valid_it.item()
            output_valid = output_valid.argmax(axis=1)
            correct_valid = torch.sum(Y_valid==output_valid)
            acc_valid[epoch*ITERATION_PER_EPOCH+iteration] = correct_valid/X_valid.size(0)
            dice_valid[epoch*ITERATION_PER_EPOCH+iteration] = calc_dice_coef(output_valid, Y_valid)
            print(f"loss train: {loss.item()}")
            print(f"loss valid: {loss_valid_it.item()}")
            print(f"dice train: {calc_dice_coef(output, Y_batch)}")
            print(f"dice valid: {calc_dice_coef(output_valid, Y_valid)}")



    plt.subplot(1, 2, 1)
    X_plt = np.arange(it)
    plt.plot(X_plt, dice_train, label="Train")
    plt.plot(X_plt, dice_valid, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Dice coef.")
    plt.title("Dice coef. over iterations")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(X_plt, loss_train, label="Train")
    plt.plot(X_plt, loss_valid, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost over iterations")
    plt.legend()
    plt.show()
    #Exercise 1.4
    #Test acc: 0.982
    #Test cost: 0.0509
    #Time: 22 min

    output_test = net(X_test.float())
    output_test = output_test.argmax(axis=1)
    dice_test = calc_dice_coef(output_test, Y_test)

    print(f"Dice test: {dice_test}")

    print("Training size: ", X_train.size())
    print("Training size: ", Y_train.size())

    # for i in range(60):
    #     plt.subplot(1, 2, 1)
    #     img = output_valid[i]
    #     imgplot = plt.imshow(img.detach())
    #     plt.subplot(1, 2, 2)
    #     img = Y_valid[i]
    #     imgplot = plt.imshow(img.detach())
    #     plt.show()
