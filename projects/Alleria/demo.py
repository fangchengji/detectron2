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

from alleria.predictor import VisualizationDemo
from alleria.config import add_alleria_config
from alleria.pseudo_label import pred_instances_to_coco_json, set_pseudo_cfg

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_alleria_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
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


def main(args, cfg, logger, pseudo_label=False):

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

    demo = VisualizationDemo(cfg)
    results = ["image_id,PredictionString\n"]
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
            # input < 10 doesn't do pseudo label
            # pseudo_label = pseudo_label & (len(args.input) > 10)
        for idx, path in tqdm.tqdm(enumerate(args.input), disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            img_size = img.shape[:2]
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            # output result
            if not pseudo_label:
                result = format_result(os.path.basename(path).split('.')[0], predictions)
                results.append(result)
            else:
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

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)

        if pseudo_label:
            # save pseudo label to json file
            json_name = args.output + '/pseudo_label.json'
            with open(json_name, 'w') as f:
                json.dump(coco_annos, f)

            # train
            from alleria.pseudo_label import register_pseudo_datasets, set_pseudo_cfg, PseudoTrainer
            # from train_net import Trainer

            # data prepare
            register_pseudo_datasets(
                image_root=os.path.dirname(args.input[0]),
                json_file=args.output + "/pseudo_label.json"
            )
            # set training cfg
            set_pseudo_cfg(cfg, len(args.input), args.output)

            # trainer
            trainer = PseudoTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
        else:
            with open('submission.csv', 'w') as f:
                f.writelines(results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    pseudo_label = True

    if pseudo_label:
        main(args, cfg, logger, pseudo_label=True)

        # after training
        cfg.defrost()
        cfg.MODEL.WEIGHTS = os.path.join(args.output, "model_final.pth")
        cfg.freeze()

    # run inference
    main(args, cfg, logger, pseudo_label=False)
