import torch
from captcha.image import ImageCaptcha
import random
import os
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
# 生成验证码图片
# num为生成验证码的个数
# count为验证码的位数
# chars是验证码中的组成的字符
# path是保存路径
# width和height是图片的宽和高
def create_data(num,count,chars,path,width,height):
    for i in range(num):
        if i % 10==0:
            print("验证码：{}".format(i))
        creater=ImageCaptcha(width=width,height=height)
        string = ""
        for j in range(count):
            random_str=random.choice(chars)
            string=random_str+string
        img = creater.generate_image(string)
        #添加干扰点
        creater.create_noise_dots(img,'#000000',4,40)
        # 添加干扰线
        creater.create_noise_curve(img,'#000000')
        file_name=path+string+'_'+ str(i) +'.jpg'
        img.save(file_name)

def download_data(path,count,num,chars,width,height):

    if not os.path.exists(path):
        os.makedirs(path)
    create_data(num,count,chars,path,width,height)


class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform, characters):
        self.file_list = list()  # 保存每个训练数据的路径
        # 使用os.listdir，获取data_dir中的全部文件
        files = os.listdir(data_dir)
        for file in files:
            path = os.path.join(data_dir, file)
            self.file_list.append(path)
        # 将数据转换对象transform保存到类中
        self.transform = transform

        # 创建一个字符到数字的字典
        self.char2int = {}
        # 在创建字符到数字的字典时，使用外界传入的字符集characters
        for i, char in enumerate(characters):
            self.char2int[char] = i

    def __len__(self):
        return len(self.file_list)

    # 函数传入索引index，函数应当返回与该索引对应的数据和标签
    # 通过dataset[i]，就可以获取到第i个样本了
    def __getitem__(self, index):
        file_path = self.file_list[index]  # 获取数据的路径
        # 打开文件，并使用convert('L')，将图片转换为灰色
        # 不需要通过颜色来判断验证码中的字符，转为灰色后，可以提升模型的鲁棒性
        image = Image.open(file_path).convert('L')
        # 使用transform转换数据，将图片数据转为张量数据
        image = self.transform(image)
        # 获取该数据图片中的字符标签
        label_char = os.path.basename(file_path).split('_')[0]

        # 在获取到该数据图片中的字符标签label_char后
        label = list()
        for char in label_char:  # 遍历字符串label_char
            # 将其中的字符转为数字，添加到列表label中
            label.append(self.char2int[char])
        # 将label转为张量，作为训练数据的标签
        label = torch.tensor(label, dtype=torch.long)
        return image, label  # 返回image和label

width=160
height=60
train_data_path="./data/train/"
test_data_path="./data/test/"
train_num=3000
test_num =100
batch_size=32
epochs=40
count=4
learning_rate = 0.01
device = torch.device("cpu")
chars='0123456789abcdefghijklmnopqrstuvwxyz'
classes=len(chars)*count

download_data(train_data_path,count,train_num,chars,width,height)
download_data(test_data_path,count,test_num,chars,width,height)
