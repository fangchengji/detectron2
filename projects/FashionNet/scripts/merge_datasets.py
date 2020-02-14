import os
import json

# for validation datasets not merged automatically

root_dir = '/data/fangcheng.ji/datasets/deepfashion2/validation'
anno_names = ["filtered_14k.json",
              "filtered_user_commodity_14k.json",
              "crop_500.json"]
anno_files = [os.path.join(root_dir, name) for name in anno_names]

merge_keys = ['images', 'annotations', 'annotations2']

with open(anno_files[0], 'r') as f:
    tmp = json.loads(f.read())
    for i in range(1, len(anno_files)):
        with open(anno_files[i], 'r') as af:
            anno = json.loads(af.read())
            for k in merge_keys:
                # add bias to id to avoid the same id
                if k != 'images':
                    bias = i * 200000
                    for item in anno[k]:
                        item['id'] += bias
                tmp[k].extend(anno[k])
                print("merge {} {} from {}, total has {}".format(len(anno[k]), k, anno_files[i], len(tmp[k])))

    # # write to new json file
    json_name = os.path.join(root_dir, 'merged_v1.json')
    with open(json_name, 'w') as ff:
        json.dump(tmp, ff)

