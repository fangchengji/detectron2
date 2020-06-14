#!/usr/bin/env python3
# @Time    : 8/5/20 7:53 PM
# @Author  : fangcheng.ji
# @FileName: convert_pth.py

import torch

weight = '/data/fangcheng.ji/detectron2/projects/Alleria/output/faster_cascade_RS_101_9x_enhance_0.25/model_final.pth'
out = '/data/fangcheng.ji/detectron2/projects/Alleria/output/faster_cascade_RS_101_9x_enhance_0.25/best.pth'

checkpoint = torch.load(weight)
state_dict = checkpoint['model']
new_checkpoint = {}
for key, value in state_dict.items():
    # print(key)
    if key in ['pixel_mean', 'pixel_std']:
        print(key, value)
    # if 'conv' in key or 'se' in key:
    #     if 'weight' in key:
    #         key = key.replace('weight', 'conv.weight')
    #     elif 'bias' in key:
    #         key = key.replace('bias', 'conv.bias')
    #     print(f'modify: {key}')
    # new_checkpoint[key] = value

# new_checkpoint['model'] = checkpoint['model']
# print(new_checkpoint)
# torch.save(new_checkpoint, out)