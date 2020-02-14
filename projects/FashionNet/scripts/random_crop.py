import os
import glob
import json
import random
from PIL import Image

#root_dir = '/Users/fangcheng.ji/Documents/datasets/deepfashion2/train'
root_dir = '/data/fangcheng.ji/datasets/deepfashion2/validation'
anno_file = os.path.join(root_dir, 'deepfashion2_coco_all_shop.json')
generate_nums = 500


def create_category2():
    categories2 = []
    categories2.append({
        'id': 1,
        'name': "commodity",
        'supercategory': "fashion",
    })
    categories2.append({
        'id': 2,
        'name': "model",
        'supercategory': "fashion"
    })
    categories2.append({
        'id': 3,
        'name': "detail",
        'supercategory': "fashion"
    })
    categories2.append({
        'id': 4,
        'name': "specification",
        'supercategory': "fashion"
    })

    return categories2


def random_crop_detail(annos, total_nums):
    num_bboxes = len(annos)
    assert num_bboxes > 0
    print("annotations num {}".format(num_bboxes))

    crop_images = []
    crop_annos = []
    crop_annos2 = []
    print("start generating {} crop pictures...".format(total_nums))
    while total_nums > 0:
        idx = random.randint(0, num_bboxes - 1)
        anno = annos[idx]

        img_id = anno['image_id']
        bbox = anno['bbox']
        area = anno['area']
        if area < 50000:          # bbox is too small
            continue
        cx = [random.randint(bbox[0], bbox[0] + bbox[2]) for _ in range(0, 2)]
        cy = [random.randint(bbox[1], bbox[1] + bbox[3]) for _ in range(0, 2)]
        x1, x2 = min(cx), max(cx)
        y1, y2 = min(cy), max(cy)
        w, h = x2 - x1, y2 - y1
        crop_box = [x1, y1, w, h]       # x, y, w, h
        crop_area = w * h
        max_edge, min_edge = max(w, h), min(w, h)
        # constraint by area and 16:9
        if crop_area < 0.15 * area \
                or crop_area > 0.8 * area \
                or max_edge > 1.78 * min_edge \
                or max_edge < 250:
            continue

        image_name = str(img_id).zfill(6) + '.jpg'
        image_path = os.path.join(root_dir, 'image', image_name)
        if not os.path.isfile(image_path):
            print("image {} is not exist".format(image_path))
            continue

        image = Image.open(image_path)
        image = image.crop((x1, y1, x2, y2))    # left, top, right, bottom
        # resize
        max_size = 500
        if max_edge < max_size:
            if w > h:
                factor = float(max_size) / float(w)
                h = int(h * factor)
                new_size = (max_size, h)
            else:
                factor = float(max_size) / float(h)
                w = int(factor * w)
                new_size = (w, max_size)
            # TODO: don't resize, just put it in a new pic.
            #  According to SSD augmentation method
            image = image.resize(new_size, Image.BICUBIC)

        crop_image_id = 300000 + total_nums
        crop_image_name = str(crop_image_id).zfill(6) + '.jpg'
        # crop image info
        w, h = image.size
        crop_images.append({
            "coco_url": "",
            "date_captured": "",
            "file_name": crop_image_name,
            "flickr_url": "",
            "id": crop_image_id,
            "license": 0,
            "width": w,
            "height": h
        })

        # crop object
        if crop_area > area / 2:
            crop_anno = anno
            # TODO: generate bbox by segementation
            crop_anno['bbox'] = [0, 0, w, h]     # should be generated by segmentation
            crop_anno['area'] = w * h
            crop_anno['image_id'] = crop_image_id
            # TODO: crop keypoints
            del crop_anno['num_keypoints']
            del crop_anno['keypoints']
            if 'segmentation' in crop_anno:
                del crop_anno['segmentation']
            crop_annos.append(crop_anno)

        # crop anno2
        crop_annos2.append({
            'image_id': crop_image_id,
            'id': total_nums,
            'category2_id': 3,
            'part': 0,
            'toward': 0
        })

        # crop image
        crop_image_path = os.path.join(root_dir, 'crop', crop_image_name)
        image.save(crop_image_path)

        total_nums -= 1

    return crop_images, crop_annos, crop_annos2


with open(anno_file, 'r') as f:
    temp = json.loads(f.read())
    crop_images, crop_annos, crop_annos2 = random_crop_detail(temp["annotations"], generate_nums)

    categories2 = create_category2()
    dataset = {
        "info": {},
        "licenses": [],
        "images": crop_images,
        "annotations": crop_annos,
        "annotations2": crop_annos2,
        "categories": temp["categories"],
        "categories2": categories2
    }

    # # write to new json file
    json_name = os.path.join(root_dir, 'deepfashion2_crop_test.json')
    with open(json_name, 'w') as ff:
        json.dump(dataset, ff)

