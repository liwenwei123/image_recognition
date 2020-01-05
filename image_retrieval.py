#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from flask import Flask

import flask
import numpy as np
import time
import json
import base64
import urllib
import torch
from torchvision import models
from torch.utils.data import *
import torch.nn as nn
import os
import random
from torchvision import transforms
from flask import current_app
import pickle
from PIL import Image
from torchvision import transforms
import requests
import cv2
import time
import requests

app = Flask(__name__)

root_dir = './data/'
feat_files = [root_dir+x for x in['gallery_feat_hard_resnet50.pkl']]
model_path = './models/resnet50_best.pth'
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cpu"
num_classes = 200
k = 20
feat_len = 2048

def load_model():
    model_ft = models.resnet50(pretrained=True)
    num_fcin = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_fcin, num_classes)
    model_ft.load_state_dict(torch.load(model_path))
    model_ft = model_ft.to(device)
    model_ft.eval()
    return model_ft

def load_gallery():
    # {'img_path': 'test_crop/stationery/200/easy/200_stationery_4053.jpg',
    #  'label': 199,
    #  'feat': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32),
    #  'pred': 199
    # }
    feat_gallery = []
    for f in feat_files:
        pickle_file=open(f,"rb")
        feat_gallery += pickle.load(pickle_file)
        pickle_file.close()

    gallery_matrix = []
    for g in feat_gallery:
        gallery_matrix.append(g['feat'])
    
    gallery_matrix = np.array(gallery_matrix)
    gallery_mod = np.linalg.norm(gallery_matrix, axis=1)

    return feat_gallery, gallery_matrix, gallery_mod

def dump_gallery(dump_file, f_path):
    if os.path.isfile(f_path):
        os.rename(f_path, f_path+'.bak')
    pickle_file=open(f_path,'wb')
    pickle.dump(dump_file, pickle_file)
    pickle_file.close()

model = load_model()
feat_gallery, gallery_matrix, gallery_mod = load_gallery()

transform = transforms.Compose([
                transforms.Resize([256,256]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

def get_query_ret(query, gallery):
    query = query.reshape([-1, feat_len])
    query_mod = np.linalg.norm(query, axis=1)
    cos_sim = np.dot(query, gallery_matrix.T) / np.dot(query_mod.reshape([-1, 1]), gallery_mod.reshape([1, -1]))
    sim_index = np.argsort(cos_sim, axis=1)[:, -k:][:, ::-1]

    results = []
    for j in range(k):
        g_index = sim_index[0][j]
        g = gallery[g_index]
        results.append({
            'gallery': root_dir + g['img_path'],
            'gallery_label':g['label'],
            'dist':cos_sim[0][g_index]
        })
    return results

def req_img_matching(request):
    url = 'http://10.108.136.51:8886/multiple/match'
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url=url, headers=headers, data=json.dumps(request))
        return json.loads(response.content.decode())
    except:
        pass
    return {'error': 'No Response From Images Matching Service.'}

@app.route("/retrieval", methods=['GET','POST'])
def user_request_processor():
    print("get a query:")
    enter = time.time()
    response = {
        'query': None,
        'topk': None,
        'message': None
    }

    if flask.request.method == 'POST':
        org_img = flask.request.json['userPic']
        if org_img is None:
            response['message'] = "error: No query image."
            return flask.jsonify(response)

        try:
            if org_img.startswith('http'):
                resp = urllib.request.urlopen(org_img)
                img = np.asarray(bytearray(resp.read()),dtype="uint8")

            else:
                img = base64.b64decode(org_img)
                img = np.fromstring(img, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = img[...,::-1]
        
        except Exception as e:
            response['message'] = "{}".format(str(e))
            return flask.jsonify(response)

        img = Image.fromarray(img)
        img = transform(img)
        img = img.unsqueeze(0)

        query_feat = []
        def hook(module, input, output):
            for x in input[0].data.cpu().numpy():
                query_feat.append(x)

        handle = model.fc.register_forward_hook(hook)
        img = img.to(device)

        t_load_img = time.time()
        print("    load img: %.3fs" % (t_load_img - enter))
        out = model(img)
        handle.remove()
        query_feat = query_feat[0]

        t_get_feat = time.time()
        print("    get features: %.3fs" % (t_get_feat - t_load_img))
        l_dist = get_query_ret(query_feat, feat_gallery)
        topk_imgs = [x['gallery'] for x in l_dist]
        response['query'] = org_img
        response['topk'] = topk_imgs
        t_get_results = time.time()
        print("    get retrieval results: %.3fs" % (t_get_results - t_get_feat))
        response = req_img_matching(response)
        print("    get matching result: %.3fs" % (time.time() - t_get_results))
        response['retrieval'] = topk_imgs
        return flask.jsonify(response)
