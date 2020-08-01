# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog

#from alleria.predictor import VisualizationDemo
from alleria.predictor import TTAPredictor, VisualizationTool
from alleria.config import add_alleria_config
from alleria.pseudo_label import pred_instances_to_coco_json, \
    register_pseudo_datasets, set_pseudo_cfg, PseudoTrainer
from alleria.ensemble_model import merge_multi_predictions

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(config_file, input_dir, output_dir):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_alleria_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.INPUT_DIR = input_dir
    cfg.OUTPUT_DIR = output_dir
    # Set score_threshold for builtin models
    # cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def format_result(image_id, predictions):
    res = image_id + ','
    if 'instances' in predictions:
        predictions = predictions['instances'].to('cpu')
        boxes = predictions.pred_boxes.tensor.numpy().astype(np.int32) if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy() if predictions.has("scores") else None
        for score, box in zip(scores, boxes):
            res += f"{score: .4f} {box[0]} {box[1]} {box[2] - box[0]} {box[3] - box[1]} "
    res += '\n'
    return res


def inference(cfgs, logger, pseudo_label=False):
    # pseudo label
    pseudo_threshold = 0.4
    coco_annos = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "wheat", "supercategory": "wheat", "skeleton": []}],
    }
    instance_id = 0

    vis_result = False
    vis_tool = VisualizationTool(cfgs[0])

    predictors = []
    for cfg in cfgs:
        predictors.append(TTAPredictor(cfg))

    results = ["image_id,PredictionString\n"]
    if cfgs[0].INPUT_DIR:
        img_list = glob.glob(cfgs[0].INPUT_DIR + "/*.jpg")
        assert img_list, "The input path(s) was not found"
        for idx, path in tqdm.tqdm(enumerate(img_list)):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            img_size = img.shape[:2]
            start_time = time.time()
            predictions = []
            for predictor in predictors:
                predictions.append(predictor(img))

            # wbf
            merged_boxes = merge_multi_predictions(predictions, img_size, nms_threshold=0.6)
            predictions = {"instances": merged_boxes}

            # output result
            if not pseudo_label:
                result = format_result(os.path.basename(path).split('.')[0], predictions)
                results.append(result)
            else:
                if 'instances' in predictions:
                    instances = predictions["instances"].to('cpu')
                    img_info, annos, instance_id = pred_instances_to_coco_json(
                        instances, path, idx, instance_id, img_size, pseudo_threshold
                    )
                    if img_info is not None and annos is not None:
                        coco_annos["images"].append(img_info)
                        coco_annos["annotations"].extend(annos)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if vis_result:
                vis_img = vis_tool(img, predictions)
                vis_img.save(os.path.join(cfgs[0].OUTPUT_DIR, os.path.basename(path)))

        if pseudo_label:
            # save pseudo label to json file
            json_name = cfgs[0].OUTPUT_DIR + '/pseudo_label.json'
            with open(json_name, 'w') as f:
                json.dump(coco_annos, f)
        else:
            with open(cfgs[0].OUTPUT_DIR + '/submission.csv', 'w') as f:
                f.writelines(results)

def pseudo_label_train(cfg, out_folder):
    # data prepare
    all_datasets = DatasetCatalog.list()
    if "wheat_coco_pseudo" not in all_datasets:
        register_pseudo_datasets(
            "wheat_coco_pseudo",
            image_root=cfg.INPUT_DIR,
            json_file=cfg.OUTPUT_DIR + '/pseudo_label.json'
        )
    # set training cfg
    if cfg.INPUT_DIR:
        img_list = glob.glob(cfg.INPUT_DIR + "/*.jpg")
    out_dir = os.path.join(cfg.OUTPUT_DIR, out_folder)
    if not os.path.exists(out_dir):
        os.mkdirs(out_dir)
    set_pseudo_cfg(cfg, len(img_list), os.path.join(cfg.OUTPUT_DIR, out_folder))

    # trainer
    trainer = PseudoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    config_file1 = "configs/qfl_atss_ensemble.yaml"
    config_file2 = "configs/qfl_atss_RS_101_FPN_3x_b8_anchor6.yaml"

    input_dir = "/data/fangcheng.ji/datasets/wheat/pseudo_test"
    output_dir = "/data/fangcheng.ji/datasets/wheat/pseudo_test_out"

    cfg1 = setup_cfg(config_file1, input_dir, output_dir)
    cfg2 = setup_cfg(config_file2, input_dir, output_dir)
    cfgs = [cfg1, cfg2]

    pseudo_label = False

    if pseudo_label:
        inference(cfgs, logger, pseudo_label=True)

        pseudo_label_train(cfgs[0], out_folder="model0")
        pseudo_label_train(cfgs[1], out_folder="model1")

        # after training
        for i, cfg in enumerate(cfgs):
            cfgs[i].defrost()
            cfgs[i].MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
            cfgs[i].freeze()

    # run inference
    inference(cfgs, logger, pseudo_label=False)
