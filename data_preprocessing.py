import pandas as pd
import numpy as np
import os
import json
import imageio
import matplotlib.pyplot as plt
from PIL import Image

data_path = '../../../retail_product_checkout/'

def get_text(file):
    f = open(file, 'r')
    content = f.read()
    f.close()
    return content

def get_crop_imgs(phase):
    cfg_file = data_path + 'instances_' + phase + '2019.json'
    instance_json = get_text(cfg_file)
    instance_dict = json.loads(instance_json)

    categories = pd.DataFrame(instance_dict['categories'])
    raw_Chinese_name_df = pd.DataFrame(instance_dict['__raw_Chinese_name_df'])
    images = pd.DataFrame(instance_dict['images'])
    annotations = pd.DataFrame(instance_dict['annotations'])
    categories.rename(columns={'id':'category_id', 'name': 'category_name'}, inplace=True)
    images.rename(columns={'id': 'image_id'}, inplace=True)
    t1 = raw_Chinese_name_df.merge(categories, on='category_id', how='left')
    t2 = images.merge(annotations, on='image_id', how='left')
    data = t2.merge(t1, on='category_id',how='left')

    crop_path = "./data/" + phase + '_crop/'
    if not os.path.isdir(crop_path):
        os.mkdir(crop_path)
    for i in range(data.shape[0]):
        row = data.iloc[i]
        skuclass = getattr(row, 'sku_class')
        cateory_id = getattr(row, 'category_id')
        if not os.path.isdir(crop_path + skuclass):
            os.mkdir(crop_path + skuclass)
        if not os.path.isdir(crop_path + skuclass + '/' + str(cateory_id)):
            os.mkdir(crop_path + skuclass + '/' + str(cateory_id))
            if phase in ['test', 'val']:
                os.mkdir(crop_path + skuclass + '/' + str(cateory_id) + '/easy')
                os.mkdir(crop_path + skuclass + '/' + str(cateory_id) + '/medium')
                os.mkdir(crop_path + skuclass + '/' + str(cateory_id) + '/hard')
        crop_size = getattr(row, 'bbox')
        file_name = getattr(row, 'file_name')
        t = Image.open(data_path + phase + '2019/' + file_name)
        t = t.crop((crop_size[0], crop_size[1], crop_size[0]+crop_size[2], crop_size[1] + crop_size[3]))
        if phase is 'train':
            t.save(crop_path + skuclass + '/' + str(cateory_id) + '/' + getattr(row, 'sku_name') + '_' +str(getattr(row, 'id')) + '.jpg')
        else:
            t.save(crop_path + skuclass + '/' + str(cateory_id) + '/' + getattr(row, 'level') + '/' + getattr(row, 'sku_name') + '_' +str(getattr(row, 'id')) + '.jpg')
        if i % 5000 == 0:  
            print("%s: %d / %d" % (phase, i, data.shape[0]))

if not os.path.isdir("./data"):
    os.mkdir("./data")
# get_crop_imgs("train")
# get_crop_imgs("val")
get_crop_imgs("test")