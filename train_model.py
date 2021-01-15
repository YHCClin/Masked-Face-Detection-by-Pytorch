# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
writer = SummaryWriter("./runs/train")

from tqdm import tqdm
import argparse

import Net.net_model as net
import Data_Loader.Dataset as data
from utils.config import Config

img_root = './agedb_30_masked/images'
train_txt = './agedb_30_masked/lables_train.txt'

batch_size = 1


# 获取训练设备
def default_service():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train(epoch):
    device = default_service()
    print("device: %s" % str(device))
    module = net.GoogLeNet()
    module.train()

    writer.add_graph(module, (torch.randn(1,3,300,300))) # 绘制模型图
    # module.to(device)
    # 优化器
    optimizer = optim.Adam(module.parameters(), lr=0.001, weight_decay=1e-8)
    # 载入训练数
    train_dataset = data.myDataset(img_dir=img_root, img_txt=train_txt, transform=data.transform)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    for Epoch in range(epoch):  # 开始训练
        all_correct_num = 0
        cur_correct_num = 0
        for ii, (img, label) in enumerate(tqdm(train_dataloader)):
            img = Variable(img)
            label = Variable(label)  # 将Tensor用Variable包装下，Variable支持Tensor的几乎一切操作
            # img.to(device)
            # label.to(device)
            output = module(img)  # 前馈计算
            optimizer.zero_grad()
            # print(output, label)
            loss = loss_func(output, label)
            writer.add_scalar('loss', loss.item(), global_step=ii)
            # print(loss.item())
            loss.backward()  # 反向传播
            _, predict = torch.max(output, 1)  # 按列取最大值
            cur_correct_num += sum(predict == label.data.item()).item()
            if ii % 300 == 299:
                writer.add_scalar('acc', cur_correct_num * 1.0 / 300, global_step=(ii+1)/300) # 每100个样本计算一次精度
                cur_correct_num = 0
            correct_num = sum(predict == label.data.item())  # 累加预测正确的样本（以一个batch为单位）
            all_correct_num += correct_num.data.item()  # 单轮（Epoch）预测的正确样本数

            optimizer.step()  # 优化器
        Accuracy = all_correct_num * 1.0 / (len(train_dataset))  # 计算本轮（Epoch）正确率
        print('Epoch ={0},all_correct_num={1},Accuracy={2}'.format(Epoch, all_correct_num, Accuracy))
        torch.save(module, './models/mask_detection.pkl')  # 保存整个模型


def main(argv):
        train(argv.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train argparse')
    parser.add_argument('--epoch', '-ep',type=int, help='epoch 训练轮数, 必要参数')
    args = parser.parse_args()
    main(args)
