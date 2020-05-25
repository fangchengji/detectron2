#!/usr/bin/env python3
# @Time    : 23/5/20 6:25 PM
# @Author  : fangcheng.ji
# @FileName: csv2coco.py

import pandas as pd
import numpy as np
import re
import json
from random import shuffle

DIR_INPUT = '/data/fangcheng.ji/datasets/wheat'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
train_df.shape

train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

image_ids = train_df['image_id'].unique()
shuffle(image_ids)

valid_ids = image_ids[:int(len(image_ids) * 0.07)]
train_ids = image_ids[int(len(image_ids) * 0.07):]

valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]


def convert_coco(df_dataset):
    dataset = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    dataset['categories'].append({
        'id': 1,
        'name': "wheat",
        'supercategory': "wheat",
        'skeleton': []
    })

    image_ids = []
    for i in range(0, len(df_dataset)):
        image_id = df_dataset.iloc[i]['image_id']
        width = df_dataset.iloc[i]['width']
        height = df_dataset.iloc[i]['height']
        if image_id not in image_ids:
            image_ids.append(image_id)
            dataset['images'].append({
                'coco_url': '',
                'date_captured': '',
                'file_name': str(image_id) + '.jpg',
                'flickr_url': '',
                'id': int(len(image_ids) - 1),
                'license': 0,
                'width': int(width),
                'height': int(height)
            })

        x = df_dataset.iloc[i]['x']
        y = df_dataset.iloc[i]['y']
        w = df_dataset.iloc[i]['w']
        h = df_dataset.iloc[i]['h']

        dataset['annotations'].append({
            'area': w * h,
            'bbox': [x, y, w, h],
            'category_id': 1,
            'id': int(i),
            'image_id': int(len(image_ids) - 1),
            'iscrowd': 0,
        })

    print(f"finished convert {len(image_ids)} images, {len(df_dataset)} instances")
    return dataset

json_name = DIR_INPUT + '/train.json'
with open(json_name, 'w') as f:
    json.dump(convert_coco(train_df), f)

json_name = DIR_INPUT + '/valid.json'
with open(json_name, 'w') as f:
    json.dump(convert_coco(valid_df), f)

print("finished!")
