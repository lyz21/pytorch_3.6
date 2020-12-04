# encoding=utf-8
"""
@Time : 2020/11/22 20:47 
@Author : LiuYanZhe
@File : train_resnet_n.py 
@Software: PyCharm
@Description: 训练从网上下载的resnet网络
"""
import torch
from torch import nn
import time
import lyzmodels.models.resnet as resnet
import lyzmodels.utils.utils_lyz as util

start_time = time.time()

# train_iter, test_iter = util.load_data_fashion_mnist(128, root='./Datasets')
train_iter, test_iter = util.load_data_cifar100(8, 64, '../Datasets')

lr = 1e-3
net = resnet.resnet152()
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss_fun = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
util.train(train_iter, test_iter, net, loss_fun, optimizer, device, 3)
print('总用时：%.1f sec' % (time.time() - start_time))
