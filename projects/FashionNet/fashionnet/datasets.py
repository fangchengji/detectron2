# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import contextlib
import os
import datetime
import sys
import time
import copy
import json
import numpy as np

from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock

from detectron2.data import MetadataCatalog, DatasetCatalog

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from collections import defaultdict

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_fashion_json"]


# ==== Predefined fashion datasets and splits for configs==========
_PREDEFINED_FASHION = {
    "fashion_train_test": (
        "deepfashion2/train_test/image",
        "deepfashion2/train_test/train_test.json",
    ),
    "fashion_train_5w": (
        "deepfashion2/train/image",
        "deepfashion2/train/filtered_5w.json",
    ),
    "fashion_train_user_commodity_5w": (
        "deepfashion2/train/image",
        "deepfashion2/train/filtered_user_commodity_5w.json",
    ),
    "fashion_train_commodity_10w": (
        "deepfashion2/train/image",
        "deepfashion2/train/filtered_commodity_10w.json",
    ),
    "fashion_train_crop_5k": (
        "deepfashion2/train/crop_5k",
        "deepfashion2/train/crop_5k.json",
    ),

    # validation datasets
    "fashion_validation_14k": (
        "deepfashion2/validation/image",
        "deepfashion2/validation/filtered_14k.json",
    ),
    "fashion_validation_merged_v1": (
        "deepfashion2/validation/image",
        "deepfashion2/validation/merged_v1.json",
    ),
    "fashion_validation_crop": (
        "deepfashion2/validation/crop",
        "deepfashion2/validation/crop_500.json",
    )
}


# All fashion categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_CLOTHES_CATEGORIES .json
CLOTHES_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "short_sleeved_shirt"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "long_sleeved_shirt"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "short_sleeved_outwear"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "long_sleeved_outwear"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "vest"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "sling"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "shorts"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "trousers"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "skirt"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "short_sleeved_dress"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "long_sleeved_dress"},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "vest_dress"},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": "sling_dress"}
]

CATEGORY2_CLASSES = ['commodity', 'model', 'detail', 'specification']
PART_CLASSES = ['top', 'down', 'whole']
TOWARD_CLASSES = ['front', 'side or back']

def _get_fashion_meta():
    thing_ids = [k["id"] for k in CLOTHES_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CLOTHES_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 13, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CLOTHES_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,

        "category2_classes": CATEGORY2_CLASSES,
        "part_classes": PART_CLASSES,
        "toward_classes": TOWARD_CLASSES,
    }

    return ret


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Fashion(COCO):
    """
    configs fashion datasets is annotated as COCO. But it contains more information than COCO.
    configs data has two annotations, named 'annotations' and 'annotations2'.
    'annotations' is the same as COCO.
    'annotations2' is extra information. It includes image id, category id, part, toward.
    """
    def __init__(self, annotation_file=None):
        self.anns2, self.cats2 = dict(), dict()
        self.imgToAnns2, self.cat2ToImgs = defaultdict(list), defaultdict(list)
        # __init__ use the createIndex method, so self variables should be defined before init.
        super().__init__(annotation_file)

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs, anns2, cats2 = {}, {}, {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        imgToAnns2, cat2ToImgs = defaultdict(list), defaultdict(list)

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'annotations2' in self.dataset:
            for ann2 in self.dataset['annotations2']:
                imgToAnns2[ann2['image_id']].append(ann2)
                anns2[ann2['id']] = ann2

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'categories2' in self.dataset:
            for cat2 in self.dataset['categories2']:
                cats2[cat2['id']] = cat2

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        if 'annotations2' in self.dataset and 'categories2' in self.dataset:
            for ann2 in self.dataset['annotations2']:
                cat2ToImgs[ann2['category2_id']].append(ann2['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

        self.anns2 = anns2
        self.imgToAnns2 = imgToAnns2
        self.cat2ToImgs = cat2ToImgs
        self.cats2 = cats2

    def getCatIds(self, catNms=[], supNms=[], catIds=[], attribute='categories'):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset[attribute]
        else:
            cats = self.dataset[attribute]
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def loadCats(self, ids=[], attribute='categories'):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if attribute == 'categories':
            cats = self.cats
        elif attribute == 'categories2':
            cats = self.cats2
        else:
            raise Exception('Attribute not supported!')

        if _isArrayLike(ids):
            return [cats[id] for id in ids]
        elif type(ids) == int:
            return [cats[ids]]


def load_fashion_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        fashion_api = Fashion(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(fashion_api.getCatIds())
        cat2_ids = sorted(fashion_api.getCatIds(attribute='categories2'))
        cats = fashion_api.loadCats(cat_ids)
        cats2 = fashion_api.loadCats(cat2_ids, attribute='categories2')

        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        classification_classes = [c["name"] for c in sorted(cats2, key=lambda x: x["id"])]
        meta.classification_classes = classification_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

        id2_map = {v: i for i, v in enumerate(cat2_ids)}
        meta.classification_dataset_id_to_contiguous_id = id2_map

    # sort indices for reproducible results
    img_ids = sorted(fashion_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = fashion_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [fashion_api.imgToAnns[img_id] for img_id in img_ids]
    anns2 = [fashion_api.imgToAnns2[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns, anns2))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + \
               ["category2_id", "part", "toward"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list, anno2_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)

        # add annotations2 to record
        clses = []
        for anno in anno2_dict_list:
            assert anno["image_id"] == image_id
            # assign the key and value to cls
            cls = {key: anno[key] for key in ann_keys if key in anno}
            if id2_map:
                cls['category2_id'] = id2_map[cls['category2_id']]
            clses.append(cls)

        record["annotations"] = objs
        record['annotations2'] = clses
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def register_fashion_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_fashion_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="fashion", **metadata
    )


def register_all_fashion(root="datasets"):
   for key, (image_root, json_file) in _PREDEFINED_FASHION.items():
       # Assume pre-defined datasets live in `../datasets`.

       register_fashion_instances(
           key,
           _get_fashion_meta(),
           os.path.join(root, json_file) if "://" not in json_file else json_file,
           os.path.join(root, image_root),
       )


# Register the fashion dataset under "detectron2/datasets"
# sudo mount --bind [your_absolute_dataset_path] [detectron2/datasets]
_DATASETS_ROOT = "../../datasets"
register_all_fashion(_DATASETS_ROOT)


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_fashion_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
