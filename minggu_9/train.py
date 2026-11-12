import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from Utils.getData import getImageLabel
from Models.CNN import SimpleCNN




def main():
    BATCH_SIZE = 32
    EPOCH = 100
    LEARNING_RATE = 0.001

    train_loader = DataLoader(getImageLabel(folder="./Dataset"), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(getImageLabel(folder="./Dataset"), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(input_dim=200, input_c=3, output=5, hidden_dim=128, dropout=0.5, device='cpu')
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    loss_train_all, loss_test_all = [], []
    for epoch in range(EPOCH):
        train_loss = 0
        test_loss = 0
        model.train()
        for batch, (src, trg) in enumerate(train_loader):
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #
        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)
            loss = loss_function(pred, trg)
            test_loss += loss.detach().numpy()

        loss_train_all.append(train_loss / len(train_loader))
        loss_test_all.append(test_loss / len(test_loader))
        print("epoch = ", epoch+1,"train loss = ", train_loss / len(train_loader), ", test loss = ", test_loss / len(test_loader))

        if (epoch+1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / len(train_loader),
            }, "./SimpleCNN_" + str(epoch+1) + ".pt")

    plt.plot(range(EPOCH), loss_train_all, color="#931a00", label='Training')
    plt.plot(range(EPOCH), loss_test_all, color="#3399e6", label='Testing')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./training.png")



if __name__ == "__main__":
    main()
