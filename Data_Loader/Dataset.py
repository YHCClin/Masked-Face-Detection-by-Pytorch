import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch
import torch.utils.data as Data
import Data_Loader.Dataset as data
import cv2

img_root = "../agedb_30_masked/images"
train_txt = "../agedb_30_masked/lables.txt"

# 加载单张图片
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))


# 重载数据集类
class myDataset(Data.DataLoader):
    def __init__(self, img_dir, img_txt, transform=None, loader=default_loader):
        # 由于数据集很大，采用小批量加载到内存中
        img_list = []  # 存储图像名称的列表
        img_labels = []  # 存储标签的列表

        fp = open(img_txt, 'r')  # 只读模式加载txt文件，文件中每行的内容为图片文件名以及标签
        for line in fp.readlines():  # 每次读取文件中的一行(即一个文件名)
            img_list.append(line.split()[0])  # 文件名添加到list中
            label = line.split()[1]
            img_labels.append(label)
        self.imgs = [os.path.join(img_dir, file) for file in img_list]  # 得到图片的相对路径
        self.labels = img_labels
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index], dtype=np.int64))
        img = self.loader(img_path)  # 加载图片并将图片转为rgb 3通道Tensor
        if self.transform is not None:
            try:
                img = self.transform(img)  # 数据处理
            except:
                print('Cannot transform image: {}'.format(img_path))
        return img, label


transform = transforms.Compose([
    transforms.Resize(300),  # 图像缩小
    # transforms.CenterCrop(128),  # 中心剪裁
    # transforms.RandomHorizontalFlip(),  # 依概率p水平翻转
    transforms.ToTensor(),  # 转tensor 并归一化
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # 标准化
                         std=[0.5, 0.5, 0.5])
])

if __name__ == "__main__":
    train_dataset = data.myDataset(img_dir=img_root, img_txt=train_txt, transform=data.transform)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for ii, (img, label) in enumerate(train_dataloader):
        print(img.size())
        print(label)
        img = img[0, 0, :, :]
        img = np.array(img)
        cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
