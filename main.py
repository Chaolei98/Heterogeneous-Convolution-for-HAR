import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.utils.data as Data
from collections import Counter
import os
import argparse
from HC_model import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=512, help='Batch size')
parser.add_argument('--lr', type=int, default=5e-4, help='Learning rate')
parser.add_argument('--epoch', type=int, default=200, help='Epoch')
args = parser.parse_args()

#load data
train_x = torch.from_numpy(np.load('/mnt/experiment/hcl/Datasets/pamap2/x_train.npy')).float()
train_y = torch.from_numpy(np.load('/mnt/experiment/hcl/Datasets/pamap2/y_train.npy')).long()
val_x = torch.from_numpy(np.load('/mnt/experiment/hcl/Datasets/pamap2/x_val.npy')).float()
val_y = torch.from_numpy(np.load('/mnt/experiment/hcl/Datasets/pamap2/y_val.npy')).long()

train_x = torch.unsqueeze(train_x, 1)
val_x = torch.unsqueeze(val_x, 1)

num_classes = len(Counter(train_y.tolist()))
len_train, len_val = len(train_y),  len(val_y)

train_dataset = Data.TensorDataset(train_x, train_y)
val_dataset = Data.TensorDataset(val_x, val_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=True)


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 50))
    optimizer.param_groups[0]['lr'] = lr

model = Resnet_HC(input_channel=1, num_classes=num_classes)
model.cuda()
print(model)

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    # print('LR:',optimizer.param_groups[0]['lr'])
    train_loss = 0
    train_num = 0
    model.train()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        loss = loss_f(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = torch.max(output, 1)[1].cpu().numpy()
        label = y.cpu().numpy()
        train_num += (pred==label).sum()
    train_acc = train_num / len_train
    print('Train Epoch:{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch, train_loss/len(train_loader), train_acc),end='||')

def val():
    val_loss = 0
    val_num = 0
    model.eval()

    for step, (x, y) in enumerate(val_loader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        loss = loss_f(output, y)

        val_loss += loss.item()

        pred = torch.max(output, 1)[1].cpu().numpy()
        label = y.cpu().numpy()
        val_num += (pred==label).sum()
    val_acc = val_num / len_val
    print('Val Loss:{:.4f} Val Acc:{:.4f}'.format(val_loss/len(val_loader), val_acc))


if __name__ == '__main__':
    for epoch in range(args.epoch):
        train(epoch)
        val()
