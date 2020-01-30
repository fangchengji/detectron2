# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from importlib import import_module

import torch


def import_obj(module_path):
    """
    :param module_path: e.g. apps.demo.src.transforms.transform.build_transforms_demo
    :return:
    """
    module_name_list = module_path.split('.')
    module_name = '.'.join(module_name_list[:-1])
    func_name = module_name_list[-1]
    module = import_module(module_name)
    if hasattr(module, func_name):
        transform_module = getattr(module, func_name)
    else:
        raise ValueError("module " + module_name + " has no attribute " + func_name)
    return transform_module


def import_class(module_path):
    return import_obj(module_path)


def import_def(module_path):
    return import_obj(module_path)


if torch._six.PY3:
    import importlib
    import importlib.util
    import sys


    # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    def import_file(module_name, file_path, make_importable=False):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if make_importable:
            sys.modules[module_name] = module
        return module
else:
    import imp


    def import_file(module_name, file_path, make_importable=None):
        module = imp.load_source(module_name, file_path)
        return module
