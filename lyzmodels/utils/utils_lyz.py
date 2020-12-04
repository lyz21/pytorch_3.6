# encoding=utf-8
"""
@Time : 2020/11/22 11:10 
@Author : LiuYanZhe
@File : utils_lyz.py
@Software: PyCharm
@Description: 调用torchvision中的模型时的方法
"""
import torchvision
import sys
import torch
import time
import torchvision.datasets as dsets
import torchvision.transforms as trancforms


def load_data_lyz(batch_size, resize, root='../Datasets/ceramic'):
    """
    加载自己数据的方法
    :param batch_size: 批量大小
    :param resize: 更改图像大小
    :param root: 文件路径
    :return: 训练数据和测试数据
    """
    # 原图像
    transform1 = trancforms.Compose([
        torchvision.transforms.Resize(size=(resize, resize)),
        torchvision.transforms.ToTensor(),
    ])
    # 进行变换的操作
    transform2 = trancforms.Compose([
        # torchvision.transforms.RandomHorizontalFlip(),  # 随机旋转图片
        torchvision.transforms.RandomCrop(resize),  # 随机切分图像
        torchvision.transforms.Resize(size=(resize, resize)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图片，默认0.5
        torchvision.transforms.ToTensor(),
    ])

    trainData1 = dsets.ImageFolder(root + '/train', transform=transform1)
    testData1 = dsets.ImageFolder(root + '/test', transform=transform1)
    trainData2 = dsets.ImageFolder(root + '/train', transform=transform2)
    testData2 = dsets.ImageFolder(root + '/test', transform=transform2)
    trainData3 = dsets.ImageFolder(root + '/train', transform=transform2)
    trainData4 = dsets.ImageFolder(root + '/train', transform=transform2)
    trainData5 = dsets.ImageFolder(root + '/train', transform=transform2)
    trainData = torch.utils.data.ConcatDataset([trainData1, trainData2, trainData3, trainData4, trainData5])
    testData = torch.utils.data.ConcatDataset([testData1, testData2])
    # print(trainData.imgs)
    train_iter = torch.utils.data.DataLoader(dataset=trainData, shuffle=True, batch_size=batch_size)
    test_iter = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size)
    show_img(train_iter, show_num=2)
    return train_iter, test_iter


def load_data_flower(batch_size, resize, root='../Datasets/ceramic_flower2'):
    """
    加载自己数据的方法
    :param batch_size: 批量大小
    :param resize: 更改图像大小
    :param root: 文件路径
    :return: 训练数据和测试数据
    """
    transform1 = trancforms.Compose([
        torchvision.transforms.Resize(size=(resize, resize)),
        # trancforms.ToTensor(),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化图片
    ])
    transform2 = trancforms.Compose([
        # torchvision.transforms.RandomHorizontalFlip(),  # 随机旋转图片
        # torchvision.transforms.CenterCrop(resize / 2),  # 从中心切分图像
        # torchvision.transforms.RandomCrop(resize),  # 随机切分图像
        torchvision.transforms.Resize(size=(resize, resize)),
        torchvision.transforms.RandomHorizontalFlip(p=1),  # 随机水平翻转图片，默认0.5
        torchvision.transforms.ToTensor(),
        # trancforms.ToTensor(),  # 转Tensor并归一化
        # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化图片
    ])

    trainData1 = dsets.ImageFolder(root + '/train', transform=transform1)
    testData1 = dsets.ImageFolder(root + '/test', transform=transform1)
    trainData2 = dsets.ImageFolder(root + '/train', transform=transform2)
    testData2 = dsets.ImageFolder(root + '/test', transform=transform2)
    # trainData3 = dsets.ImageFolder(root + '/train', transform=transform2)
    # trainData4 = dsets.ImageFolder(root + '/train', transform=transform2)
    # trainData5 = dsets.ImageFolder(root + '/train', transform=transform2)
    trainData = torch.utils.data.ConcatDataset([trainData1, trainData2])
    testData = torch.utils.data.ConcatDataset([testData1, testData2])
    # print(trainData.imgs)
    train_iter = torch.utils.data.DataLoader(dataset=trainData, shuffle=True, batch_size=batch_size)
    test_iter = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size)
    show_img_n_batch(train_iter, nrow=16, show_batch_num=6)
    return train_iter, test_iter


def load_data_cifar100(batch_size, resize=None, root='./Datasets'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    """训练方法"""
    loss_list = []
    loss_test_list = []
    acc_test_list = []
    acc_train_list = []
    if str(device) == 'cuda':
        net = net.cuda(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        batch_count = 0
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            if batch_count % 1000 == 0:
                print('train ', batch_count, 'batch')
        test_acc, test_loss, _ = evaluate_accuracy(test_iter, net, loss)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        loss_list.append(train_l_sum / batch_count)
        loss_test_list.append(test_loss)
        acc_test_list.append(test_acc)
        acc_train_list.append(train_acc_sum / n)

    return loss_list, loss_test_list, acc_train_list, acc_test_list


def evaluate_accuracy(data_iter, net, loss, device=None):
    """
    计算准确度
    :param data_iter: 数据集
    :param net: 网络
    :param loss: 损失函数
    :param device: cuda or cpu
    :return:
    """
    # 如果没指定device就使用net的device
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    if str(device) == 'cuda':
        net = net.cuda(device)
    test_loss_sum, acc_sum, n = 0.0, 0.0, 0
    batch_num = 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # if batch_num >= 3000:
                #     break
                net.eval()  # 评估模式, 这会关闭dropout
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                if batch_num == 0:
                    y_hat_torch = y_hat.argmax(dim=1)
                else:
                    y_hat_torch = torch.cat((y_hat_torch, y_hat.argmax(dim=1)), dim=0)
                l = loss(y_hat, y)
                test_loss_sum += l.cpu().item()
                acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
                batch_num += 1
            else:
                print('evaluate_accuracy wrong')
            n += y.shape[0]
    return acc_sum / n, test_loss_sum / batch_num, y_hat_torch


def show_img(pic_iter, show_num=1, nrow=2, is_label=False, fig_size=None, pre_y=None):
    """
    绘制图片和标签的方法。
    :param pic_iter:读取的得到的train_iter和test_iter。（batch_size,channel,weight,height）
    :param show_num: 展示的batch数目
    :param nrow: 每行的数目
    :param is_label: 是否展示标签。如果选择True，下面的参数必须初始化
    :param fig_size: 图像尺寸大小
    :param pre_y: 预测的标签
    :return: 无返回值
    """
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import numpy as np
    num = 0  # 计算输出几个batch
    padding = 4  # 图片之间填充大小
    for x, y in pic_iter:  # 遍历iter里面的每一个batch
        if num >= show_num:  # 如果输出大于设定的数目，跳出循环
            break
        # 将图片展示成网格
        img = make_grid(x, nrow=nrow, padding=padding, normalize=True, range=None, scale_each=False, pad_value=1)
        np_img = img.numpy()  # 转为numpy格式
        if is_label:  # 如果展示标签
            fig_num_x, fig_num_y = 0, 0  # 初始化行和列号，用于计算标签位置
            for i in range(len(y)):  # 遍历该batch内的每一个标签
                if fig_num_x >= nrow:  # 换到下一行
                    fig_num_x = 0
                    fig_num_y += 1
                # 绘制标签
                plt.text(fig_size / 4 + fig_size * fig_num_x, (fig_size + padding) * fig_num_y + padding / 2,
                         'Label Pre:' + str(pre_y[i].item()) + ',Ture:' + str(y[i].item()))
                fig_num_x += 1  # 更新列号
        # plt.axis("off")  # 不显示坐标尺寸
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()
        num += 1


def show_img_n_batch(pic_iter, nrow=10, show_batch_num=-1, is_label=False, fig_size=64, pre_y=None):
    """
    绘制n个batch的图片和标签的方法。
    :param pic_iter: 读取的得到的train_iter和test_iter。（batch_size,channel,weight,height）
    :param nrow: 每行的数目
    :param show_num: 展示的batch数目
    :param is_label: 是否展示标签。如果选择True，下面的参数必须初始化
    :param fig_size: 图像尺寸大小
    :param pre_y: 预测的标签
    :return: 无返回值
    """
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import numpy as np
    padding = int(fig_size / 9)  # 图片之间填充大小
    flag = 0  # 标志位，初始化X_torch和y_torch
    num = 0  # 统计图片展示batch个数
    plt.figure(figsize=(nrow * 3, nrow * 3 * 2 / 5))  # （宽，高）
    for x, y in pic_iter:  # 遍历iter里面的每一个batch
        if flag == 0:
            x_torch = x
            y_torch = y
            flag = 1
        else:
            x_torch = torch.cat((x_torch, x), dim=0)
            y_torch = torch.cat((y_torch, y), dim=0)
        num += 1
        if show_batch_num != -1:
            if num >= show_batch_num:
                break

    if is_label:  # 如果展示标签
        fig_num_x, fig_num_y = 0, 0  # 初始化行和列号，用于计算标签位置
        for i in range(len(y_torch)):  # 遍历该batch内的每一个标签
            if fig_num_x >= nrow:  # 换到下一行
                fig_num_x = 0
                fig_num_y += 1
            # 绘制标签
            if pre_y[i].item() == y_torch[i].item():
                color = 'b'
            else:
                color = 'r'
            plt.text(fig_size / 4 + (fig_size + padding) * fig_num_x,
                     (fig_size + padding) * fig_num_y + padding * 3 / 4,
                     'PRE:' + str(pre_y[i].item()) + ',TURE:' + str(y_torch[i].item()), size=int(fig_size / 10),
                     color=color)
            fig_num_x += 1  # 更新列号

    img = make_grid(x_torch, nrow=nrow, padding=padding, normalize=True, range=None, scale_each=False, pad_value=1)
    np_img = img.numpy()  # 转为numpy格式

    plt.axis("off")  # 不显示坐标尺寸
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    load_data_flower(4, 128, root='../Datasets/ceramic_flower3')
