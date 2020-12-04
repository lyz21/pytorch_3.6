# encoding=utf-8
"""
@Time : 2020/11/22 20:19 
@Author : LiuYanZhe
@File : train_torch_Net.py
@Software: PyCharm
@Description: 调用torchvision中的模型训练
若要使用，需要修改：
1. 超参数output_num必须修改，为输出的类别个数
2. 保存参数集的名称，加载数据集的路径
2. 若要更改模型，后面更该模型最后一层的参数也需要修改。因为模型最后全连接层的个数不同
3. pre函数使用前，需要确认预测输出类别、网络模型、最后一层的更改、加载的参数集
"""
import lyzmodels.utils.utils_lyz as util
import torchvision.models as models
import torch
from torch import nn
import time
import matplotlib.pyplot as plt


def train_pre():
    start_time = time.time()

    # 参数
    lr = 1e-4
    num_epochs = 15
    fig_size = 256
    batch_num = 8
    output_num = 8

    # 模型
    model = models.resnet152(pretrained=True)
    # model = models.resnet18(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.vgg11(pretrained=True)
    # model = models.inception_v3(pretrained=True)
    print(model)
    print('=' * 20)
    # 冻结参数
    for name, param in model.named_parameters():
        if 'fc' not in name:  # 除最后一个fc层外，其他参数全部冻结
            param.requires_grad = False
    # 更改最后一层
    # model.fc = torch.nn.Linear(512, output_num)  # resnet18,512为resnet18的固定值，100为预测输出个数
    model.fc = torch.nn.Linear(2048, output_num)  # resnet152
    # model.fc = torch.nn.Linear(1024, output_num)  # googlenet
    print(model)

    # 加载参数
    # model.load_state_dict(torch.load('../net_parameter/resnet_4_acc0.85iter100.pt'))

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
    # train_iter, test_iter = util.load_data_cifar100(8, 64, root='../Datasets')  # batch-size=8，图片大小为64
    # train_iter, test_iter = util.load_data_lyz(1, 64, root='../Datasets/ceramic')
    # train_iter, test_iter = util.load_data_lyz(2, 128, root='../Datasets/ceramic_flower')
    train_iter, test_iter = util.load_data_flower(batch_num, fig_size, root='../Datasets/ceramic_flower3_t')
    # train_iter, test_iter = util.load_data_flower(batch_num, fig_size, root='../Datasets/ceramic_4')
    # train_iter, test_iter = util.load_data_flower(batch_num, fig_size, root='../Datasets/ceramic4_y')
    # train_iter, test_iter = util.load_data_flower(batch_num, fig_size, root='../Datasets/ceramic5')

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

    # 保存模型参数
    # torch.save(model.state_dict(), '../net_parameter/resnet152_8_acc' + str(test_acc) + 'iter' + str(num_epochs) + '.pt')
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


def pre(path):
    start_time = time.time()
    # 参数
    fig_size = 256
    batch_num = 4
    # cpu or GPU (CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型
    model = models.resnet152(pretrained=True)
    # model = models.resnet18(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.vgg11(pretrained=True)
    # model = models.inception_v3(pretrained=True)
    # 更改最后一层
    # model.fc = torch.nn.Linear(1024, 4)  # googlenet
    model.fc = torch.nn.Linear(2048, 4)  # resnet152
    # 损失函数
    loss_fun = nn.CrossEntropyLoss()
    # 加载参数
    model.load_state_dict(torch.load(path))
    # 数据
    train_iter, test_iter = util.load_data_flower(batch_num, fig_size, root='../Datasets/ceramic_flower3_t')
    # 预测
    test_acc, test_loss, y_hat = util.evaluate_accuracy(test_iter, model, loss_fun, device)
    # 绘制图片
    util.show_img_n_batch(test_iter, nrow=20, show_batch_num=100, is_label=True, fig_size=fig_size, pre_y=y_hat)
    print('测试准确度：', test_acc)
    print('总用时：%.1f sec' % (time.time() - start_time))


if __name__ == '__main__':
    train_pre()
    # pre('../net_parameter/resnet152_6_acc0.9745762711864406iter50.pt')
