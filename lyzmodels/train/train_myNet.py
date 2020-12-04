# encoding=utf-8
"""
@Time : 2020/11/22 20:34 
@Author : LiuYanZhe
@File : train_myNet.py 
@Software: PyCharm
@Description: 训练自己的网络，训练古陶瓷图片
"""
import lyzmodels.utils.utils_lyz as util
import lyzmodels.models.oneconnet as one_net
import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

start_time = time.time()

in_channel, out_class_num, figsize = 3, 2, 64
# 模型
model = one_net.one_conv_net(in_channel, out_class_num, figsize)
model.fc = nn.Linear(int((figsize - 4) / 2) ** 2 * 32, out_class_num)

print(model)
# 参数
lr = 1e-3
num_epochs = 20
# 优化函数与损失函数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # 冻结参数不更新
loss_fun = nn.CrossEntropyLoss()
# loss_fun = nn.NLLLoss()  # 适合用log_softmax的
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
# 数据
# train_iter, test_iter = util.load_data_lyz(1, 64, root='../Datasets/ceramic')
# train_iter, test_iter = util.load_data_lyz(2, figsize, root='../Datasets/ceramic_flower')
train_iter, test_iter = util.load_data_flower(4, figsize, root='../Datasets/ceramic_flower2')
print('len(train_iter):', len(train_iter))
# 训练
loss_list, loss_train_test_list, acc_train_list, acc_test_list = util.train(train_iter, test_iter, model, loss_fun,
                                                                            optimizer,
                                                                            device,
                                                                            num_epochs)
print('训练总用时：%.1f sec' % (time.time() - start_time))

test_acc, test_loss, _ = util.evaluate_accuracy(test_iter, model, loss_fun, device)
print('测试准确度：', test_acc)
print('总用时：%.1f sec' % (time.time() - start_time))

# 绘制训练图像
plt.subplot(2, 1, 1)
plt.title('Test Accuracy = ' + str(test_acc))
plt.plot(range(0, len(acc_train_list)), acc_train_list, label='train_acc')
plt.plot(range(0, len(acc_test_list)), acc_test_list, label='test_acc')
plt.xlabel('iterations number')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(2, 1, 2)
plt.title('Test Loss = ' + str(test_loss))
plt.plot(range(0, len(loss_list)), loss_list, label='train_loss')
plt.plot(range(0, len(loss_train_test_list)), loss_train_test_list, label='test_loss')
plt.legend()
plt.xlabel('iterations number')
plt.ylabel('loss')
plt.show()
