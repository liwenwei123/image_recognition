import torch
from torchvision import models
from torch.utils.data import *
import torch.nn as nn
import os
import random
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle

image_dir = './data/'
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 200
k = 50

#Dataloader
class dataset(Dataset):
    def get_data_dict(self, phase):
        classes = os.listdir(self.root_dir + self.imgs_path)
        ret = dict()
        for cls in classes:
            if cls == 'personal_hygiene':
                continue
            cls_dict = dict()
            for cid in os.listdir(self.root_dir + self.imgs_path + cls):
                if phase in ['test', 'val']:
                    level_dict = dict()
                    for level in os.listdir(self.root_dir + self.imgs_path + cls + '/' + cid):
                        level_dict[level] = os.listdir(self.root_dir + self.imgs_path + cls + '/' + cid + '/' + level)
                    cls_dict[int(cid)] = level_dict
                else:
                    cls_dict[int(cid)] = os.listdir(self.root_dir + self.imgs_path + cls + '/' + cid)
            ret[cls] = cls_dict
        return ret
    
    def get_images(self, phase='train', level='all'):
        imgs_file = self.get_data_dict(phase)
        imgs = []
        labels = []
        for key in imgs_file.keys():
            for key_id in imgs_file[key].keys():
                if phase in ['train']:
                    for item in imgs_file[key][key_id]:
                        imgs.append(self.imgs_path + key + '/' + str(key_id) + '/' + item)
                        labels.append(key_id-1)
                else:
                    if level in ['easy', 'medium', 'hard']:
                        for item in imgs_file[key][key_id][level]:
                            imgs.append(self.imgs_path + key + '/' + str(key_id) + '/' + level + '/' + item)
                            labels.append(key_id-1)
                    else:
                        for llevel in imgs_file[key][key_id].keys():
                            imgs.append(self.imgs_path + key + '/' + str(key_id) + '/' + llevel + '/' + item)
                            labels.append(key_id-1)
                        
        random.seed(2019)
        random.shuffle(imgs)
        random.seed(2019)
        random.shuffle(labels)
        return imgs, labels
    
    def __init__(self, root_dir='./data/', phase='train', level='all'):
        self.phase = phase
        self.root_dir = root_dir
        self.imgs_path = phase + '_crop/'
        self.imgs, self.labels = self.get_images(phase, level)
        self.data_transforms = {
            'train': transforms.Compose([
                #transforms.RandomRotation(60, expand=True),
                transforms.RandomResizedCrop(224, scale=(0.2, 0.5)),
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
        img = Image.open(image_dir + img_path)
        img = self.data_transforms[self.phase](img)
        return img_path, img, self.labels[idx]
    
    def __len__(self):
        return len(self.imgs)

def load_model():
    model_ft = models.resnet50(pretrained=True)
    num_fcin = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_fcin, num_classes)
    model_ft.load_state_dict(torch.load('./models/resnet50_best.pth'))
    model_ft = model_ft.to(device)
    return model_ft

def get_features(phase, data_loaders):
    model.eval()
    
    ret = []
    pred = []
    features = []
    def hook(module, input, output):
        for x in input[0].data.cpu().numpy():
            features.append(x)
    handle = model.fc.register_forward_hook(hook)
    
    for step, (imgs_path, inputs, labels) in enumerate(data_loaders[phase]):
        for i in range(len(imgs_path)):
            ret.append({'img_path': imgs_path[i], 'label': labels[i].item()})
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        for out in torch.argmax(outputs, 1).data.cpu().numpy():
            pred.append(out)
        if (step+1) % 100 == 0:
            print("%d / %d" % ((step+1), len(data_loaders[phase])))
        
    handle.remove()
    model.train()

    for i in range(len(features)):
        ret[i]['feat'] = features[i]
        ret[i]['pred'] = pred[i]
    return ret

def generate_gallery(dif, ret):
    feat_file = './data/gallery_feat_' + dif + '_resnet50_no_personal_hygiene.pkl'
    pickle_file=open(feat_file,'wb')
    pickle.dump(ret, pickle_file)
    pickle_file.close()

def get_matrix(data):
    data_matrix = []
    for g in data:
        data_matrix.append(g['feat'])
    
    data_matrix = np.array(data_matrix)
    data_mod = np.linalg.norm(data_matrix, axis=1)
    return data_matrix, data_mod

def count_topk_acc(q_ret, topk):
    acc_count = [0] * len(topk)
    
    for q in q_ret:
        g_labels = [x['gallery_label'] for x in q['gallery_dist']]
        for i in range(len(topk)):
            if q['query_label'] in g_labels[:topk[i]]:
                acc_count[i] += 1
    return [x/len(q_ret) for x in acc_count]

def test(dif):
    data_loaders = {
        'test': DataLoader(dataset(image_dir, 'test', dif), batch_size=batch_size, shuffle=True)
    }
    dataset_sizes = {x: len(data_loaders[x].dataset) for x in ['test']}

    ret = get_features("test", data_loaders)
    generate_gallery(dif, ret);

    count = 0
    for item in ret:
        if item['pred'] == item['label']:
            count += 1
    print("pred accuracy on %s dataset: %.4f" % (dif, (count/len(ret))))

    random.shuffle(ret)
    cut = int(len(ret)/4)
    query = ret[:cut]
    gallery = ret[cut:]
    query_matrix, query_mod = get_matrix(query)
    gallery_matrix, gallery_mod = get_matrix(gallery)

    cos_sim = np.dot(query_matrix, gallery_matrix.T) / np.dot(query_mod.reshape([-1, 1]), gallery_mod.reshape([1, -1]))
    sim_index = np.argsort(cos_sim, axis=1)[:, -k:][:, ::-1]

    q_ret = []
    for i in range(len(query)):
        q = query[i]
        results = []
        for j in range(k):
            g_index = sim_index[i][j]
            g = gallery[g_index]
            results.append({
                'gallery': g['img_path'],
                'gallery_label':g['label'],
                'dist':cos_sim[i][g_index]
            })
        q_ret.append({
            'query:': q['img_path'],
            'query_label': q['label'],
            'gallery_dist': results
        })

    topk = [1, 5, 10, 20, 30, 50]
    topk_retrieval_acc = count_topk_acc(q_ret, topk)
    print("topk retrieval accrucy on %s data:" % phase)
    for i in range(len(topk)):
        print("    top%d retrieval acc: %.4f" % (topk[i], topk_retrieval_acc[i]))

model = load_model()
test("hard")
# for phase in ['easy', 'medium', 'hard']:
#     test(phase)