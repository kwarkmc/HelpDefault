import glob
import sys,os
import os.path as osp
import random
import numpy as np
import json

import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

num_epochs=30
class DefaultDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)
        #train 폴더 내의 사진의 개수 return

    def __getitem__(self, index):
        #train 이미지의 transformed 데이터와 라벨 return
        # index번째의 화상 로드
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)
        # 전이학습을 수행하기 위한 데이터의 사이즈 torch.size([3,224,224])


        if img_path.find("\\t1\\")!=-1:
            label = 1
        elif img_path.find("\\t2\\")!=-1:
            label = 2
        elif img_path.find("\\t3\\")!=-1:
            label = 3
        elif img_path.find("\\t4\\")!=-1:
            label = 4
        elif img_path.find("\\t5\\")!=-1:
            label = 5
        elif img_path.find("\\t6\\")!=-1:
            label = 6
        elif img_path.find("\\t7\\")!=-1:
            label = 7
        elif img_path.find("\\t8\\")!=-1:
            label = 8
        elif img_path.find("\\t9\\")!=-1:
            label = 9
        elif img_path.find("\\t10\\")!=-1:
            label = 10
        elif img_path.find("\\t11\\")!=-1:
            label = 11


        return img_transformed, label


def make_datapath_list():

    """
    데이터의 경로를 저장한 리스트 작성

    Parameters
    ----------
    phase : 'train' or 'val'
        훈련 데이터 또는 검증 데이터 지정
    Returns
    -------
    path_list : list
        데이터 경로를 지정한 리스트
    """

    target_path = osp.join('train/*/**.jpg')
    l_dict = {i:0 for i in range(1,12)}
    path_list=[]
    size = 10
    for path in glob.glob(target_path)*10:
        if path.find("\\t1\\")!=-1:
            if l_dict[1]<size:
                l_dict[1]+=1
            else:
                continue
        elif path.find("\\t2\\")!=-1:
            if l_dict[2]<size:
                l_dict[2]+=1
            else:
                continue
        elif path.find("\\t3\\")!=-1:
            if l_dict[3]<size:
                l_dict[3]+=1
            else:
                continue
        elif path.find("\\t4\\")!=-1:
            if l_dict[4]<size:
                l_dict[4]+=1
            else:
                continue
        elif path.find("\\t5\\")!=-1:
            if l_dict[5]<size:
                l_dict[5]+=1
            else:
                continue
        elif path.find("\\t6\\")!=-1:
            if l_dict[6]<size:
                l_dict[6]+=1
            else:
                continue
        elif path.find("\\t7\\")!=-1:
            if l_dict[7]<size:
                l_dict[7]+=1
            else:
                continue
        elif path.find("\\t8\\")!=-1:
            if l_dict[8]<size:
                l_dict[8]+=1
            else:
                continue
        elif path.find("\\t9\\")!=-1:
            if l_dict[9]<size:
                l_dict[9]+=1
            else:
                continue
        elif path.find("\\t10\\")!=-1:
            if l_dict[10]<size:
                l_dict[10]+=1
            else:
                continue
        elif path.find("\\t11\\")!=-1:
            if l_dict[11]<size:
                l_dict[11]+=1
            else:
                continue
        path_list.append(path)

    return path_list

class ImageTransform():
    """
    화상 전처리 클래스. 훈련 시, 검증 시의 동작이 다르다.
    화상 크기를 리사이즈하고 색상을 표준화한다.
    훈련 시에는 RandomResizedCrop과 RandomHorizontalFlip으로 데이터를 확장한다.

    Attributes
    ----------
    resize : int
        리사이즈 대상 화상의 크기
    mean : (R, G, B)
        각 색상 채널의 평균 값
    std : (R, G, B)
        각 색상 채널의 표준편차차
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드 지정
        """
        return self.data_transform[phase](img)


df_log={"epoch":[0]*(num_epochs+1),"epoch_loss":[0]*(num_epochs+1),"epoch_acc":[0]*(num_epochs+1)}
train_list= make_datapath_list()
val_list= make_datapath_list()
size=224
mean=(0.485,0.456,0.406)
std = (0.229,0.224,0.225)

train_dataset = DefaultDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

val_dataset = DefaultDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

batch_size=32

train_dataloader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader = data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)


dataloaders_dict = {"train":train_dataloader,"val":val_dataloader}

use_pretrained = True
net = models.vgg16(use_pretrained)

net.classifier[6]=nn.Linear(in_features=4096,out_features=12)
net.train()

criterion = nn.CrossEntropyLoss()

params_to_update_1=[]
params_to_update_2=[]
params_to_update_3=[]

update_param_names_1=['features']
update_param_names_2=['classifier.0.weight',
                      'classifier.0.bias','classifier.3.weight','classifier.3.bias']
update_param_names_3 = ['classifier.6.weight','classifier.6.bias']

for name,param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad=True
        params_to_update_1.append(param)
        print("params_to_update_1에 저장 ",name)
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2에 저장 ", name)
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3에 저장 ", name)
    else:
        param.requires_grad=False
        print("경사 계산 없음. 학습하지 않음", name)

optimizer = optim.SGD([
    {'params':params_to_update_1,'lr':1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params':params_to_update_3,'lr':1e-3}
], momentum=0.9)

def train_model(net,dataloaders_dict, criterion, optimizer, num_epochs):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("사용 장치: ", device)

    net.to(device)

    torch.backends.cudnn.benchmark=True

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-'*30)

        for phase in ['train']:
            net.train()

            epoch_loss=0.0
            epoch_corrects=0


            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs)
                    loss=criterion(outputs,labels)
                    _, preds=torch.max(outputs,1)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss +=loss.item()*inputs.size(0)
                    epoch_corrects +=torch.sum(preds==labels.data)

            epoch_loss = epoch_loss/len(dataloaders_dict[phase].dataset)
            epoch_acc=epoch_corrects.double()/len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} acc: {:.4f}'.format(phase,epoch_loss, epoch_acc))
            df_log['epoch'][epoch]=epoch
            df_log['epoch_loss'][epoch]=epoch_loss
            df_log['epoch_acc'][epoch]=int(epoch_acc)
            log = pd.DataFrame(df_log)
            log.to_csv('log_data.csv')
            save_path = './weight_fine_tunning{}epoch.pth'.format(epoch)
            torch.save(net.state_dict(),save_path)


train_model(net,dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

