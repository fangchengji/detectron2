#!/usr/bin/env python3
# @Time    : 8/5/20 7:53 PM
# @Author  : fangcheng.ji
# @FileName: convert_eff_pth.py

import torch

weight = '/root/.cache/torch/checkpoints/adv-efficientnet-b5-86493f6b.pth'

checkpoint = torch.load(weight)
new_checkpoint = {}
for key, value in checkpoint.items():
    print(key)
    if 'conv' in key or 'se' in key:
        if 'weight' in key:
            key = key.replace('weight', 'conv.weight')
        elif 'bias' in key:
            key = key.replace('bias', 'conv.bias')
        print(f'modify: {key}')
    new_checkpoint[key] = value

torch.save(new_checkpoint, weight)
