import os
import pandas as pd
import numpy as np
import torchvision
import torch
from torchvision import models, transforms
from torch import nn
from torch import optim
from torch.utils.data import *
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import copy
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

image_dir = './data/'
batch_size = 64
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_classes = 200

def get_data_dict(phase):
    data_path = image_dir + phase + '_crop/'
    classes = os.listdir(data_path)
    ret = dict()
    for cls in classes:
        cls_dict = dict()
        for cid in os.listdir(data_path + cls):
            if phase in ['test', 'val']:
                level_dict = dict()
                for level in os.listdir(data_path + cls + '/' + cid):
                    level_dict[level] = os.listdir(data_path + cls + '/' + cid + '/' + level)
                cls_dict[int(cid)] = level_dict
            else:
                cls_dict[int(cid)] = os.listdir(data_path + cls + '/' + cid)
        ret[cls] = cls_dict
    return ret

#Dataloader
class dataset(Dataset):
    def get_images(self, imgs_file, imgs_path, phase='train', level='all'):
        imgs = []
        labels = []
        for key in imgs_file.keys():
            for key_id in imgs_file[key].keys():
                if phase in ['train']:
                    for item in imgs_file[key][key_id]:
                        imgs.append(imgs_path + key + '/' + str(key_id) + '/' + item)
                        labels.append(key_id-1)
                else:
                    if level in ['easy', 'medium', 'hard']:
                        for item in imgs_file[key][key_id][level]:
                            imgs.append(imgs_path + key + '/' + str(key_id) + '/' + level + '/' + item)
                            labels.append(key_id-1)
                    else:
                        for llevel in imgs_file[key][key_id].keys():
                            imgs.append(imgs_path + key + '/' + str(key_id) + '/' + llevel + '/' + item)
                            labels.append(key_id-1)
                        
        random.seed(2019)
        random.shuffle(imgs)
        random.seed(2019)
        random.shuffle(labels)
        return imgs, labels
    
    def __init__(self, imgs_file, imgs_path, phase='train', level='all'):
        self.phase = phase
        self.imgs, self.labels = self.get_images(imgs_file, imgs_path, phase, level)
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 0.8)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize([256,256]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([256,256]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = self.data_transforms[self.phase](img)
        return img_path, img, self.labels[idx]
    
    def __len__(self):
        return len(self.imgs)

#Train
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            epoch_loss = 0
            epoch_corrects = 0
            # Iterate over data.
            last_time = time.time()
            count_inputs = 0
            for step, (_, inputs, labels) in enumerate(data_loaders[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and (step+1) % 100 == 0:
                    print('[epoch: %d, step: %d] loss: %.3f, time: %.3fs' %
                          (epoch, step+1, running_loss/100, time.time() - last_time))
                    train_writer.add_scalar('loss',
                            running_loss/100,
                            epoch * len(data_loaders['train']) + step+1)
                    running_loss = 0.0
                    running_corrects = 0
                    last_time = time.time()
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = epoch_loss / dataset_sizes[phase]
            epoch_acc = epoch_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_writer.add_scalar('epoch loss',
                            epoch_loss,
                            epoch)
                train_writer.add_scalar('acc',
                            epoch_acc,
                            epoch)
            else:
                val_writer.add_scalar('epoch loss',
                            epoch_loss,
                            epoch)
                val_writer.add_scalar('acc',
                            epoch_acc,
                            epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, './models/resnet50_best.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

train_files = get_data_dict('train')
val_files = get_data_dict('val')

data_loaders = {
    'train': DataLoader(dataset(train_files, image_dir + 'train_crop/', 'train'), batch_size=batch_size, shuffle=True),
    'val': DataLoader(dataset(val_files, image_dir + 'val_crop/', 'val', 'easy'), batch_size=batch_size, shuffle=True)
}

dataset_sizes = {x: len(data_loaders[x].dataset) for x in data_loaders.keys()}

model_ft = models.resnet50(pretrained=True)
num_fcin = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_fcin, num_classes)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.005)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

if not os.path.isdir("./runs"):
    os.mkdir("./runs")
train_writer = SummaryWriter('runs/train')
val_writer = SummaryWriter('runs/val')

if not os.path.isdir("./models"):
    os.mkdir("./models")
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)