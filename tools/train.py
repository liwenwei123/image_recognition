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
import imgaug as ia
import imgaug.augmenters as iaa
%matplotlib inline

image_dir = './data/'
batch_size = 64
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_classes = 200

def get_data_dict(data_name):
    data_path = image_dir + data_name + '_crop/'
    classes = os.listdir(data_path)
    ret = dict()
    for cls in classes:
        cls_dict = dict()
        for cid in os.listdir(data_path + cls):
            if data_name in ['test', 'val']:
                level_dict = dict()
                for level in os.listdir(data_path + cls + '/' + cid):
                    level_dict[level] = os.listdir(data_path + cls + '/' + cid + '/' + level)
                cls_dict[int(cid)] = level_dict
            else:
                cls_dict[int(cid)] = os.listdir(data_path + cls + '/' + cid)
        ret[cls] = cls_dict
    return ret

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.5), "y": (0.8, 1.5)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            #rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                #sometimes(iaa.Superpixels(p_replace=(0, 0.3), n_segments=(50, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-5, 20), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=False
        ),
        iaa.Resize(256)
    ],
    random_order=False
)

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
                        key_id_files = os.listdir(imgs_path + key + '/' + str(key_id) + '/' + level)
                        for i in range(10):
                            idx = random.randint(0, len(key_id_files)-1)
                            imgs.append(imgs_path + key + '/' + str(key_id) + '/' + level + '/' + key_id_files[idx])
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
                transforms.Lambda(lambda img: seq(image=img)),
                #transforms.ToPILImage(),
                #transforms.RandomResizedCrop(224, scale=(0.5, 0.875)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([300,300]),
                transforms.CenterCrop(256),
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
        #img = Image.open(img_path)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
        img = self.data_transforms[self.phase](img)
        return img_path, img, self.labels[idx]
    
    def __len__(self):
        return len(self.imgs)

data_loaders = {
    'train': DataLoader(dataset(train_files, image_dir + 'train_crop/', 'train'), batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(dataset(val_files, image_dir + 'val_crop/', 'val', 'easy'), batch_size=batch_size, shuffle=True)
}
dataset_sizes = {x: len(data_loaders[x].dataset) for x in ['train', 'val']}

# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# %%time
# k = 32
# plt.figure(figsize=(20, 10))
# # Get a batch of training data
# imgs_path, inputs, classes = next(iter(data_loaders['train']))

# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs[:k])

# imshow(out, title=[str(x) for x in classes.data.numpy()][:k])

# for item in imgs_path[:k]:
#     print(item)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
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
                           ( epoch * len(data_loaders['train']) + step+1) // 100)
                    running_loss = 0.0
                    running_corrects = 0
                    last_time = time.time()
                
            if phase == 'train' and scheduler is not None:
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
                torch.save(best_model_wts, './system/models/net_param_imgaug111.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.resnet50(pretrained=True)
num_fcin = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_fcin, num_classes)
model_ft = model_ft.to(device)
#model_ft.load_state_dict(torch.load("./system/models/net_param_imgaug.pth"))

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

train_writer = SummaryWriter('./runs/train')
val_writer = SummaryWriter('./runs/val')

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=200)

