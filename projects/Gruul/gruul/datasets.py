#!/usr/bin/env python3
# @Time    : 26/5/20 10:30 AM
# @Author  : fangcheng.ji
# @FileName: datasets.py

import os
import logging
import sys
import torch

from detectron2.data import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""
logger = logging.getLogger(__name__)

# ==== Predefined fashion datasets and splits for configs==========
_PREDEFINED_DATASETS = {
    "size_chart_train": "size_chart/train",
    "size_chart_val": "size_chart/val",
}


def _find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and not d.name.startswith('_')]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and not d.startswith('_')]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def _load_image_folder_data_dicts(dir, class_to_idx):
    data_dicts = []
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        folders = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        folders = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    for folder in folders:
        img_folder = os.path.join(dir, folder)
        for img in os.listdir(img_folder):
            record = {}
            record["file_name"] = os.path.join(img_folder, img)
            if not folder.startswith('_'):
                record["class"] = folder
                record["class_id"] = torch.tensor(class_to_idx[folder])
            else:
                record["class"] = "negative"
                record["class_id"] = torch.tensor(-1)

            data_dicts.append(record)

    return data_dicts


def register_image_folder_instances(name, image_root):
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
        image_root (str): directory which contains all the images.
    """
    classes, class_to_idx = _find_classes(image_root)
    metadata = {
        "class_to_idx": class_to_idx,
        "classes": classes,
        # "thing_colors": thing_colors,
    }

    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: _load_image_folder_data_dicts(image_root, class_to_idx))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_root=image_root, evaluator_type="classification", **metadata
    )


def register_all_datasets(root="datasets"):
   for key, image_root in _PREDEFINED_DATASETS.items():
       # Assume pre-defined datasets live in `../datasets`.
       register_image_folder_instances(key, os.path.join(root, image_root))


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_datasets(_root)