# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from .dataset_mapper import DatasetMapper
from .config import add_fashionnet_config
from .data_build import build_detection_test_loader, build_detection_train_loader
from .predictor import VisualizationDemo
from .fashionnet import FashionNet
from .fashion_evaluation import FashionEvaluator
from . import datasets  # ensure the datasets are registered
#from .grouped_batch_sampler import GroupedBatchSampler

#__all__ = [
#    "DatasetMapper",
#    "TrainingSampler",
#    "InferenceSampler",
#    "RepeatFactorTrainingSampler",
#    "FashionNetCOCOEvaluator",
#    "add_fashionnet_config",
#    "build_detection_train_loader",
#    "build_detection_test_loader",
#    "VisualizationDemo",
#]
__all__ = [k for k in globals().keys() if "datasets" not in k and not k.startswith("_")]
