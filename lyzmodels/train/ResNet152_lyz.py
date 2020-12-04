# encoding=utf-8
"""
@Time : 2020/12/4 17:24 
@Author : LiuYanZhe
@File : ResNet152_lyz.py 
@Software: PyCharm
@Description: 调用torchvision.models里面的ResNet152，修改，构建自己的模型。
"""
import time
import lyzmodels.utils.utils_lyz as util
import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt

class ResNet152_lyz(nn.Module):
    """
    定义自己的模型结构
    """

    def __init__(self, num_classes=10):
        """
        :param num_classes:分类类别数目
        """
        super(ResNet152_lyz, self).__init__()
        net = models.resnet152(pretrained=True)  # 定义模型，并加载参数
        net.fc = nn.Sequential()  # 最后的分类层置空
        self.feature = net  # 保存特征提取层
        self.fc = nn.Sequential(  # 定义自己的分类层
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train():
    start_time = time.time()

    # 参数
    lr = 1e-4
    num_epochs = 15
    fig_size = 256
    batch_num = 8
    output_num = 8

    # 模型
    model = ResNet152_lyz(output_num)
    print(model)
    # 冻结参数
    for name, param in model.named_parameters():
        print('layer name:',name)
        if 'fc' not in name:  # 除最后一个fc层外，其他参数全部冻结
            param.requires_grad = False

    # 优化方法
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # 冻结参数不更新
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 损失函数
    loss_fun = nn.CrossEntropyLoss()

    # cpu or GPU (CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 数据
    train_iter, test_iter = util.load_data_flower(batch_num, fig_size, root='../Datasets/ceramic_flower3_t')

    # 训练
    loss_list, loss_train_test_list, acc_train_list, acc_test_list = util.train(train_iter, test_iter, model, loss_fun,
                                                                                optimizer,
                                                                                device,
                                                                                num_epochs)

    print('训练总用时：%.1f sec' % (time.time() - start_time))

    # 预测
    test_acc, test_loss, y_hat = util.evaluate_accuracy(test_iter, model, loss_fun, device)
    print('测试准确度：', test_acc)
    print('总用时：%.1f sec' % (time.time() - start_time))

    # 绘制图片 show_batch_num大于batch总数，就是显示全部
    util.show_img_n_batch(test_iter, nrow=20, show_batch_num=100, is_label=True, fig_size=fig_size, pre_y=y_hat)
    # 绘制训练图像
    plt.subplot(2, 1, 1)
    plt.title('Accuracy = ' + str(test_acc))
    plt.plot(range(0, len(acc_train_list)), acc_train_list, label='train_acc')
    plt.plot(range(0, len(acc_test_list)), acc_test_list, label='test_acc')
    plt.xlabel('iterations number')
    plt.ylabel('accuracy')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title('Loss = ' + str(test_loss))
    plt.plot(range(0, len(loss_list)), loss_list, label='train_loss')
    plt.plot(range(0, len(loss_train_test_list)), loss_train_test_list, label='test_loss')
    plt.legend()
    plt.xlabel('iterations number')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    train()