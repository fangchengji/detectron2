#!/usr/bin/env python3
# @Time    : 31/5/20 10:59 AM
# @Author  : fangcheng.ji
# @FileName: loader.py

import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from fvcore.common.file_io import PathManager
from tabulate import tabulate
from termcolor import colored

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import log_first_n

from detectron2.data.build import get_detection_dataset_dicts, trivial_batch_collator, worker_init_reset_seed
from detectron2.data import samplers
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper

from .dataset_mapper import PlusDatasetMapper


def build_detection_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        an infinite iterator of training data
    """
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = PlusDatasetMapper(cfg, dataset, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # drop_last so the batch always have the same size
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    return data_loader


def build_detection_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = PlusDatasetMapper(cfg, dataset, is_train=False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


if __name__=="__main__":
    config_file = "/data/fangcheng.ji/detectron2/projects/Alleria/configs/fcos_R_50_FPN_1x.yaml"

    from detectron2.config import get_cfg
    from detectron2.engine import default_setup
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from datasets import register_all_datasets
    import os

    def instances_to_json(instances):
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.gt_boxes.tensor.numpy()
        if boxes.shape[1] == 4:
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        # scores = instances.scores.tolist()
        classes = instances.gt_classes.tolist()

        results = []
        for k in range(num_instance):
            result = {
                "category_id": classes[k],
                "bbox": boxes[k],
                "bbox_mode": BoxMode.XYWH_ABS,
            }

            results.append(result)
        return results

    cfg = get_cfg()
    cfg.DATALOADER.MOSAIC_AUGMENTATION = 0.0
    # mix-up augmentation
    cfg.DATALOADER.MIXUP_AUGMENTATION = 0.0
    # hsv augmentation
    cfg.DATALOADER.HSV_AUGMENTATION = 0.0

    cfg.merge_from_file(config_file)
    cfg.freeze()
    default_setup(cfg, "")

    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN)
    data_loader = build_detection_train_loader(cfg)

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    data_iter = iter(data_loader)
    for idx in range(10):
        batches = next(data_iter)
        images = [x["image"].numpy().transpose(1, 2, 0) for x in batches]
        file_names = [x["file_name"] for x in batches]
        annos = [{"annotations" : instances_to_json(x["instances"])} for x in batches]
        for image, file_name, anno in zip(images, file_names, annos):
            visualizer = Visualizer(image, metadata=meta)
            vis = visualizer.draw_dataset_dict(anno)
            fpath = os.path.join(dirname, os.path.basename(file_name))
            vis.save(fpath)


