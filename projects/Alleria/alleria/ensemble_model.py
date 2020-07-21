#!/usr/bin/env python3
# @Time    : 17/7/20 11:48 AM
# @Author  : fangcheng.ji
# @FileName: ensemble_model.py

import datetime
import logging
import time
import torch

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.config import get_cfg
from detectron2.evaluation import DatasetEvaluators

from .config import add_alleria_config
from .boxes_fusion import boxes_fusion_single_image


def get_config(config_file):
    cfg = get_cfg()
    add_alleria_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def merge_multi_predictions(outputs, image_shape, nms_threshold):
    all_boxes = []
    all_scores = []
    all_classes = []
    for output in outputs:
        # Need to inverse the transforms on boxes, to obtain results on original image
        if isinstance(output, dict) and 'instances' in output:
            output = output['instances']
        all_boxes.append(output.pred_boxes.tensor)
        all_scores.append(output.scores)
        all_classes.append(output.pred_classes)
    # all_boxes = torch.cat(all_boxes, dim=0)

    merged_instances = boxes_fusion_single_image(
        all_boxes, all_scores, all_classes, image_shape,
        nms_threshold,
        method="wbf",
        device=all_boxes[0].device,
    )
    return merged_instances


def inference_ensemble_on_dataset(models, data_loader, evaluator):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    for model in models:
        model.eval()

    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()

            outputs = []
            for model in models:
                outputs.append(model(inputs))

            res = []
            for i in range(len(outputs[0])):
                out_i = [output[i] for output in outputs]
                merged_instances = merge_multi_predictions(
                    out_i,
                    (inputs[i]['height'], inputs[i]['width']),
                    nms_threshold=0.6,
                )
                res.append({"instances": merged_instances})

            outputs = res
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results
