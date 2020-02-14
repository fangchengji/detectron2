import os
import glob
import json

root_dir = '/Users/fangcheng.ji/Documents/datasets/deepfashion2/validation'
anno_file = os.path.join(root_dir, 'deepfashion2_coco_user_14k.json')

detail_set = set()
detail_paths = glob.glob(os.path.join(root_dir, 'detail') + '/*.jpg')
for path in detail_paths:
    detail_set.add(os.path.basename(path))

commodity_set = set()
commodity_paths = glob.glob(os.path.join(root_dir, 'commodity') + '/*.jpg')
for path in commodity_paths:
    commodity_set.add(os.path.basename(path))

top_set = set()
top_paths = glob.glob(os.path.join(root_dir, 'model', 'top') + '/*.jpg')
for path in top_paths:
    top_set.add(os.path.basename(path))

down_set = set()
down_paths = glob.glob(os.path.join(root_dir, 'model', 'down') + '/*.jpg')
for path in down_paths:
    down_set.add(os.path.basename(path))

whole_set = set()
whole_paths = glob.glob(os.path.join(root_dir, 'model', 'whole') + '/*.jpg')
for path in whole_paths:
    whole_set.add(os.path.basename(path))

unknown_set = set()
unknown_paths = glob.glob(os.path.join(root_dir, 'unknown') + '/*.jpg')
for path in unknown_paths:
    unknown_set.add(os.path.basename(path))

subindex = 0
with open(anno_file, 'r') as f:
    temp = json.loads(f.read())
    annotations2 = temp['annotations2']
    print('length of annotations2 {}'.format(len(annotations2)))

    # build map
    img2anno_idx = {}
    for i, it in enumerate(temp["annotations"]):
        if it['image_id'] not in img2anno_idx:
            img2anno_idx[it['image_id']] = [i]
        else:
            img2anno_idx[it['image_id']].append(i)
    img2img_idx = {}
    for i, it in enumerate(temp["images"]):
        img2img_idx[it["id"]] = i

    filter_count = 0
    filter_annos = []
    filter_images = []
    filter_annos2 = []
    for it in annotations2:
        image_name = str(it['image_id']).zfill(6)+'.jpg'
        subindex += 1
        it['id'] = subindex

        if image_name in detail_set:
            it['category2_id'] = 3
            it['part'] = 0
            it['toward'] = 0
        elif image_name in commodity_set:
            it['category2_id'] = 1
            it['part'] = 0
            it['toward'] = 0
        elif image_name in top_set:
            it['category2_id'] = 2
            it['part'] = 1
        elif image_name in down_set:
            it['category2_id'] = 2
            it['part'] = 2
        elif image_name in whole_set:
            it['category2_id'] = 2
            it['part'] = 3
        elif image_name in unknown_set:
            it['category2_id'] = 0
            it['part'] = 0
            it['toward'] = 0
        else:
            continue

        filter_count += 1
        filter_annos2.append(it)
        filter_images.append(temp["images"][img2img_idx[it['image_id']]])
        filter_annos.extend([temp["annotations"][j] for j in img2anno_idx[it["image_id"]]])

    temp["images"] = filter_images
    temp["annotations2"] = filter_annos2
    temp["annotations"] = filter_annos
    print("Finished filter with {} results".format(filter_count))

    # write to new json file
    json_name = os.path.join(root_dir, 'deepfashion2_filtered_10w.json')
    with open(json_name, 'w') as ff:
        json.dump(temp, ff)
