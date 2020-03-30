import argparse
import os
import torch
import itertools

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.structures import ImageList
from detectron2.config import CfgNode as CN

from fashionnet.config import add_fashionnet_config


def add_export_config(cfg):
    """
    Args:
        cfg (CfgNode): a detectron2 config

    Returns:
        CfgNode: an updated config with new options that :func:`export_caffe2_model` will need.
    """
    is_frozen = cfg.is_frozen()
    cfg.defrost()
    cfg.EXPORT_CAFFE2 = CN()
    cfg.EXPORT_CAFFE2.USE_HEATMAP_MAX_KEYPOINT = False
    if is_frozen:
        cfg.freeze()
    return cfg


def setup_cfg(args):
    cfg = get_cfg()
    add_fashionnet_config(cfg)
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    # set export_onnx flag to true
    cfg.MODEL.EXPORT_ONNX = True
    cfg.MODEL.FASHIONNET.EXPORT_ONNX = True
    cfg = add_export_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def preprocess_image(cfg, batched_inputs, device, backbone_size_divisibility):
    """
    Normalize, pad and batch the input images.
    """
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(device).view(3, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std

    images = [x["image"].to(device) for x in batched_inputs]
    images = [normalizer(x) for x in images]
    images = ImageList.from_tensors(images, backbone_size_divisibility)
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model to Caffe2")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted caffe2 model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    setup_logger()
    logger = setup_logger(name="fashionnet", abbrev_name="fashionnet")
    logger.info("Command line arguments: " + str(args))

    cfg = setup_cfg(args)
    input_names = ["input"]
    # output_names = ["detection_logits", "detection_bbox_reg", "classification_logits"]
    output_names = [[f"logits_{i}", f"bbox_reg_{i}", f"ctrness_{i}"] for i in range(3, 8)]
    output_names = list(itertools.chain(*output_names))
    output_path = os.path.join(args.output, "fashionnet_fcos_v39_800.onnx")

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    # get a sample data
    # data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    # first_batch = next(iter(data_loader))
    # dummy_in = preprocess_image(
    #     cfg, first_batch, torch_model.device, torch_model.size_divisibility
    # ).tensor
    # set input size to 800 * 800
    # dummy_in = dummy_in[:, :, :800, :800]

    input_size = cfg.INPUT.SIZE
    dummy_in = torch.randn(1, 3, input_size[1], input_size[0], device=torch_model.device)

    torch.onnx.export(
        torch_model,
        dummy_in,
        output_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names)

    logger.info("Finished exporting onnx model to {}".format(output_path))


