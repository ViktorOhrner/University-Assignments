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

# class CustomDataSet():
#     def __init__(self, dir, transform=None):
#         self.DIR = dir
#
#     def __len__(self):
#         pass
#
#     def __getitem__(self, index):
#         img_path =


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.conv2_bn = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=(1, 1), padding_mode='reflect')
        self.conv3_bn = nn.BatchNorm2d(16)
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        self.tconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.convFinal_bn = nn.BatchNorm2d(8)
        self.convFinal = nn.Conv2d(8, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=(2, 2)))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=(2, 2)))
        x = F.relu(self.conv3(x))
        # print(x.size())
        x = F.relu(self.tconv1(x))
        # print(x.size())
        x = F.relu(self.tconv2(x))
        x = F.relu(self.convFinal(x))
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
    random_index = torch.randperm(X_train.size(0))

    #TEST PLOT AN IMAGE
    # img = X_train[0]
    # print(img.size())
    # imgplot = plt.imshow(img.permute(1, 2, 0))
    # plt.show()

    lr = 0.01
    batch_size = 5
    iterations = 5000
    EPOCHS = math.floor(batch_size*iterations/X_train.shape[0])
    ITERATION_PER_EPOCH = math.floor(X_train.shape[0]/batch_size)

    net = Net()
    net = net.float()
    # optimizer = optim.Adam(net.parameters(), lr)
    optimizer = optim.Adam(net.parameters(), lr)
    # optimizer = optim.SGD(net.parameters(), lr, momentum=0.9) # Try this one
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    it = EPOCHS*ITERATION_PER_EPOCH
    loss_train=torch.zeros(it)
    acc_train=torch.zeros(it)
    dice_train=torch.zeros(it)
    loss_test=torch.zeros(it)
    acc_test=torch.zeros(it)
    dice_test=torch.zeros(it)

    print(f"Epochs: {EPOCHS}")
    print(f"Iterations per epoch: {ITERATION_PER_EPOCH}")

    for epoch in range(EPOCHS):
        random_index = torch.randperm(X_train.size(0))
        for iteration in range(ITERATION_PER_EPOCH):
            print(f"Iteration: {epoch*ITERATION_PER_EPOCH+iteration}")
            X_batch = X_train[random_index[iteration*batch_size:(iteration+1)*batch_size],:,:,:]
            Y_batch = Y_train[random_index[iteration*batch_size:(iteration+1)*batch_size],:,:]
            # plt.subplot(1, 2, 1)
            # img = X_batch[0, 0]
            # imgplot = plt.imshow(img.detach())
            # plt.subplot(1, 2, 2)
            # img = Y_batch[0]
            # imgplot = plt.imshow(img.detach())
            # plt.show()
            net.zero_grad()
            output = net(X_batch.float())
            # print(output.view(batch_size, 2, 128*128).size())
            # print(Y_batch.view(batch_size, 128*128).size())
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            loss_train[epoch*ITERATION_PER_EPOCH+iteration] = loss.item()
            # img = output[0, 0]
            # imgplot = plt.imshow(img.detach())
            # plt.show()
            output = output.argmax(axis=1)
            # if ((epoch*ITERATION_PER_EPOCH+iteration)%1000)== 0:
            #     plt.subplot(1, 2, 1)
            #     img = output[0]
            #     imgplot = plt.imshow(img.detach())
            #     plt.subplot(1, 2, 2)
            #     img = Y_train[random_index[iteration]]
            #     imgplot = plt.imshow(img.detach())
            #     plt.show()

            correct = torch.sum(Y_batch==output)
            acc_train[epoch*ITERATION_PER_EPOCH+iteration] = correct/batch_size
            dice_train[epoch*ITERATION_PER_EPOCH+iteration] = calc_dice_coef(output, Y_batch)

            output_test = net(X_test.float())
            loss_test_it = criterion(output_test.squeeze(), Y_test)
            loss_test[epoch*ITERATION_PER_EPOCH+iteration] = loss_test_it.item()
            output_test = output_test.argmax(axis=1)
            correct_test = torch.sum(Y_test==output_test)
            acc_test[epoch*ITERATION_PER_EPOCH+iteration] = correct_test/X_test.size(0)
            dice_test[epoch*ITERATION_PER_EPOCH+iteration] = calc_dice_coef(output_test, Y_test)
            print(f"loss train: {loss.item()}")
            print(f"loss test: {loss_test_it.item()}")
            print(f"dice train: {calc_dice_coef(output, Y_batch)}")
            print(f"dice test: {calc_dice_coef(output_test, Y_test)}")




    X_plt = np.arange(it)
    plt.plot(X_plt, dice_train, label="Train")
    plt.plot(X_plt, dice_test, label="Test")
    plt.xlabel("Iterations")
    plt.ylabel("Dice coef.")
    plt.title("Dice coef. over iterations")
    plt.legend()
    plt.show()

    plt.plot(X_plt, loss_train, label="Train")
    plt.plot(X_plt, loss_test, label="Test")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost over iterations")
    plt.legend()
    plt.show()
    #Exercise 1.4
    #Test acc: 0.982
    #Test cost: 0.0509
    #Time: 22 min

    print(f"Accuracy: {acc_test[-1]}")
    print(f"Cost: {loss_test[-1]}")
    print(f"Dice: {dice_test[-1]}")

    for i in range(60):
        plt.subplot(2, 2, 1)
        img = output_test[i]
        imgplot = plt.imshow(img)
        plt.subplot(2, 2, 2)
        img = Y_test[i]
        imgplot = plt.imshow(img)
        plt.subplot(2, 2, 3)
        img = X_test[i, 0]
        imgplot = plt.imshow(img)
        plt.subplot(2, 2, 4)
        img = X_test[i, 1]
        imgplot = plt.imshow(img)

        plt.show()
