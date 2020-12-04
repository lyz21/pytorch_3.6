# encoding=utf-8
"""
@Time : 2020/11/22 19:36 
@Author : LiuYanZhe
@File : oneconnet.py 
@Software: PyCharm
@Description: conv-->batchnormalization->maxpool->fc
"""
import torch.nn as nn
import torch.nn.functional as F


class one_conv_net(nn.Module):
    def __init__(self, in_channel=3, out_class_num=3, fig_size=32):
        super(one_conv_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 5)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.out_fig_size = (((fig_size - 4) / 2) - 4) / 2
        # self.fc = nn.Linear(int(self.out_fig_size) ** 2 * 64, out_class_num)
        self.fc = nn.Linear(30 * 30 * 32, out_class_num)

    def forward(self, X):
        feature1 = self.pool1(self.act1(self.conv1(X)))
        # feature2 = self.pool2(self.act2(self.conv2(feature1)))
        x = self.fc(feature1.view(X.shape[0], -1))
        # return F.log_softmax(x, dim=1)
        return x
