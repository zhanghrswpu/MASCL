# config.py
import os
from torchvision import transforms

use_gpu=True
gpu_name=1

#pre_model=os.path.join('D:\\LYF\\paper2\\contrastrival_learning\\model','model_stage1_epoch100.pth')

#save_path="D:\\Python\\pycharm\\adv_code\\paper2\\My-lab\\paper2\\contrastrival_learning\\model"

#服务器
save_path = "/root/models"

train_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
