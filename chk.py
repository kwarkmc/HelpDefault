import glob
import sys,os
import os.path as osp
import random
import numpy as np
import json
import time
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

size=224
mean=(0.485,0.456,0.406)
std = (0.229,0.224,0.225)

def b_func():
    path = os.path.join("**.jpg")
    p_list = []
    for i in glob.glob(path):
        p_list.append(i)

    return p_list

def interpret(p_list):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = models.vgg16()
    net.classifier[6] = nn.Linear(in_features=4096, out_features=12)
    net.load_state_dict(
        torch.load("C:/Users/kwarkmc/Desktop/DL/HelpDefault/weight_fine_tunning24epoch.pth",map_location=device))
    net.to(device)
    net.eval()
    img = Image.open(p_list)
    transformer = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    img = transformer(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    ans = net(img)
    return int(torch.argmax(ans))
