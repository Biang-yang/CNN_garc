# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:12:20 2020

@author: 盒子先生
"""

from __future__ import print_function, division
# 解决OpenMP库冲突，让画图正常
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 强制设置matplotlib为交互式后端，确保弹窗能出来
import matplotlib
matplotlib.use('TkAgg')

import time
import numpy as np
import torch as t

import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

use_gpu = t.cuda.is_available()
if use_gpu:
    print('gpu可用')
else:
    print('gpu不可用')

epochs = 50 # 训练次数
batch_size = 6 # 批处理大小
num_workers = 0 # 多线程的数目
model = 'model.pt' # 把训练好的模型保存下来

'''
# 对加载的图像作归一化处理， 全部改为[32x32x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''
# 训练集 → 做数据增强
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(15),           # 随机旋转 ±15 度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试集 → 不增强，保持原样
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 在训练集中，shuffle必须设置为True，表示次序是随机的
# trainset = datasets.ImageFolder(root='datasets/train/', transform=data_transform)
trainset = datasets.ImageFolder(root='datasets/train/', transform=train_transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# testset = datasets.ImageFolder(root='datasets/test/', transform=data_transform)
testset = datasets.ImageFolder(root='datasets/test/', transform=test_transform)
testloader = t.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

classes = ('cardboard', 'glass', 'metal', 'paper','plastic','trash')

class Net(nn.Module):
    def __init__(self):
        # 卷积
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # 池化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 全连接
        self.fc1 = nn.Linear(in_features=400,out_features=120)
        self.dropout1 = nn.Dropout(0.3)  # 新增：随机丢弃50%的神经元---1
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.dropout2 = nn.Dropout(0.3)  # 新增：随机丢弃50%的神经元---2
        self.fc3 = nn.Linear(in_features=84, out_features=6)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        # 改变张量维度
        x = x.view(x.size(0),-1)

        # 全连接部分（关键修改：加Dropout）
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # 经过Dropout---3
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # 经过Dropout---4
        x = self.fc3(x)
        return x

def train():
    net = Net()
    if use_gpu:
        net = net.cuda()

    print("开始训练")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_count = []
    test_accuracy_count = []
    train_accuracy_count = []
    diff_count = []

    t.set_num_threads(8)
    start = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        net.train()

        # 训练
        for i, data in enumerate(trainloader, 0):
            # 输入数据
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # 梯度清零
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 更新参数
            optimizer.step()

            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()

        # 改的地方1.0
        # 每个 epoch 结束，统计一次平均 loss
        loss_count.append(running_loss / len(trainloader))

        # 改的地方2.0
        # 每个 epoch 结束，统计测试集准确率
        correct = 0
        total = 0
        net.eval()
        with t.no_grad():
            for data in testloader:
                images, labels = data
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = net(images)
                _, predicted = t.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        test_accuracy_count.append(test_acc)

        # 每个 epoch 结束，统计训练集准确率
        train_correct = 0
        train_total = 0
        with t.no_grad():
            for data in trainloader:
                images, labels = data
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = net(images)
                _, predicted = t.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_accuracy_count.append(train_acc)
        diff_count.append(train_acc - test_acc)

        print('[%d] loss: %.3f  test_acc:%.3f  train_acc:%.3f'
              % (epoch+1, loss_count[-1], test_acc, train_acc))

    # 改的地方3.0
    # 训练全部结束后统一画图 + 保存图片
    end = time.time()
    print("训练完毕！总耗时：%d 秒" % (end - start))

    plt.figure('CNN_Loss')
    plt.plot(loss_count, label='Loss')
    plt.legend()
    plt.savefig('CNN_Loss.png', dpi=300)
    plt.show()

    plt.figure('CNN_Test_Accuracy')
    plt.plot(test_accuracy_count, label='Test_Accuracy')
    plt.legend()
    plt.savefig('CNN_Test_Accuracy.png', dpi=300)
    plt.show()

    plt.figure('CNN_Train_Accuracy')
    plt.plot(train_accuracy_count, label='Train_Accuracy')
    plt.legend()
    plt.savefig('CNN_Train_Accuracy.png', dpi=300)
    plt.show()

    plt.figure('CNN_Diff_Accuracy')
    plt.plot(diff_count, label='Diff_Accuracy')
    plt.legend()
    plt.savefig('CNN_Diff_Accuracy.png', dpi=300)
    plt.show()

    t.save(net, model)

def test():
    correct = 0 # 预测正确的图片数
    total = 0 #总共的图片数
    print("开始检测")
    net = t.load(model)
    net.eval()

    with t.no_grad():
        for data in testloader:
            images, labels = data
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = t.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('测试集准确率: %.1f %%' % (100 * correct / total))

train()
test()