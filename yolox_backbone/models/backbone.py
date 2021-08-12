#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .yolo_pafpn import YOLOPAFPN
from .yolo_fpn import YOLOFPN
from ..utils.utils import download_from_url
from ..utils.torch_utils import intersect_dicts 

import torch
import torch.nn as nn
import os

model_dict = {"yolox-s": {"depth": 0.33, "width": 0.50, "depthwise": False}, 
              "yolox-m": {"depth": 0.67, "width": 0.75, "depthwise": False}, 
              "yolox-l": {"depth": 1.0, "width": 1.0, "depthwise": False}, 
              "yolox-x": {"depth": 1.33, "width": 1.25, "depthwise": False}, 
              "yolox-nano": {"depth": 0.33, "width": 0.25, "depthwise": True}, 
              "yolox-tiny": {"depth": 0.33, "width": 0.375, "depthwise": False},
              "yolox-darknet53": {"depth": 53}
              }

model_urls = {"yolox-s": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth", 
              "yolox-m": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth", 
              "yolox-l": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth",
              "yolox-x": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth",
              "yolox-nano": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth",  
              "yolox-tiny": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny_32dot8.pth",
              "yolox-darknet53" : "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.pth"
             }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_model(model_name, pretrained=False, out_features=["P3", "P4", "P5"]):
    model_name = model_name.lower()
    if not model_name in model_dict.keys():
        raise RuntimeError(f"Unknown model {model_name}")
    
    out_features = list(set(out_features))
    if not all(out_feature in ["P3", "P4", "P5"] for out_feature in out_features):
        raise RuntimeError(f'The values in out_features must be one of ["P3", "P4", "P5"].')

    Backbone = YOLOFPN if model_name == "yolox-darknet53" else YOLOPAFPN
    
    model = Backbone(**model_dict[model_name], out_features=out_features)
    
    if pretrained:
        filename = os.path.join(BASE_DIR, model_name + ".pth")
        if not os.path.isfile(filename):
            download_from_url(url=model_urls[model_name], filename=filename)
        
        assert os.path.isfile(filename), f"{model_name} weights file doesn't exist"
        
        chkpt = torch.load(filename)
        state_dict = chkpt["model"]
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if "backbone." in k:
                # (1) k = backbone.backbone.* or (2) k = backbone.*
                k = k[9:]
                # (1) k = backbone.* or (2) k = *
            backbone_state_dict[k] = v
        pretrained_model_state_dict = intersect_dicts(backbone_state_dict, model.state_dict())
        
        assert len(pretrained_model_state_dict) == len(model.state_dict())
        model.load_state_dict(pretrained_model_state_dict, len(model.state_dict()))
    return model

def list_models():
    return [key for key in model_dict.keys()]
