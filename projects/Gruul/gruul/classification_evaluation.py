#!/usr/bin/env python3
# @Time    : 29/5/20 3:38 PM
# @Author  : fangcheng.ji
# @FileName: classification_evaluation.py

import logging
import torch
import itertools
from collections import OrderedDict

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.logger import create_small_table


class ClassificationEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._distributed = distributed
        self._output_dir = output_dir

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if "class_id" in output:
                pred_class = output["class_id"].to(self._cpu_device)
                self._predictions.append({"pred_class_id": pred_class, "gt_class_id": input["class_id"]})

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[ClassificationEvaluator] Did not receive valid predictions.")
            return {}

        pred_nums = [0] * len(self._metadata.classes)
        gt_nums = [0] * len(self._metadata.classes)
        correct_nums = [0] * len(self._metadata.classes)
        for p in self._predictions:
            if p['gt_class_id'] >= 0:
                gt_nums[p['gt_class_id']] += 1
                if p['gt_class_id'] == p['pred_class_id']:
                    correct_nums[p['gt_class_id']] += 1
            if p['pred_class_id'] >= 0:
                pred_nums[p['pred_class_id']] += 1
        result = {}
        eps = 0.00001
        for i, cls in enumerate(self._metadata.classes):
            idx = self._metadata.class_to_idx[cls]
            acc = correct_nums[idx] / (pred_nums[idx] + eps)
            recall = correct_nums[idx] / (gt_nums[idx] + eps)
            result.update({
                cls + "_acc": acc,
                cls + "_recall": recall
            })
        total_acc = sum(correct_nums) / (sum(pred_nums) + eps)
        total_recall = sum(correct_nums) / (sum(gt_nums) + eps)
        result.update({
            "total_acc": total_acc,
            "total_recall": total_recall
        })
        self._logger.info(
            "Evaluation results for classification: \n" + create_small_table(result)
        )
        results = OrderedDict()
        results["classification"] = result
        return results

