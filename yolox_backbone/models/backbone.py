#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .yolo_pafpn import YOLOPAFPN
from ..utils.utils import download_from_url
from ..utils.torch_utils import intersect_dicts 

import torch
import torch.nn as nn
import os

model_dict = {"yolox-s": {"depth_scaling_factor": 0.33, "width_scaling_factor": 0.50, "depthwise": False}, 
              "yolox-m": {"depth_scaling_factor": 0.67, "width_scaling_factor": 0.75, "depthwise": False}, 
              "yolox-l": {"depth_scaling_factor": 1.0, "width_scaling_factor": 1.0, "depthwise": False}, 
              "yolox-x": {"depth_scaling_factor": 1.33, "width_scaling_factor": 1.25, "depthwise": False}, 
              "yolox-nano": {"depth_scaling_factor": 0.33, "width_scaling_factor": 0.25, "depthwise": True}, 
              "yolox-tiny": {"depth_scaling_factor": 0.33, "width_scaling_factor": 0.375, "depthwise": False}
              }

model_urls = {"yolox-s": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth", 
              "yolox-m": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth", 
              "yolox-l": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.pth",
              "yolox-x": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_x.pth",
              "yolox-nano": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth",  
              "yolox-tiny": "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny_32dot8.pth"
             }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_model(model_name, pretrained=False):
    model_name = model_name.lower()
    if not model_name in model_dict.keys():
        raise RuntimeError(f"Unknown model {model_name}")
    
    model = YOLOXBackbone(**model_dict[model_name])
    
    if pretrained:
        filename = os.path.join(BASE_DIR, model_name + ".pth")
        if not os.path.isfile(filename):
            download_from_url(url=model_urls[model_name], filename=filename)
        
        assert os.path.isfile(filename), f"{model_name} weights file doesn't exist"
        chkpt = torch.load(filename)
        pretrained_model_state_dict = chkpt["model"]
        pretrained_model_state_dict = intersect_dicts(pretrained_model_state_dict, model.state_dict())
        
        model.load_state_dict(pretrained_model_state_dict)
        
    return model

def list_models():
    return [key for key in model_dict.keys()]

class YOLOXBackbone(nn.Module):
    """
    YOLOX Backbone model module.
    """

    def __init__(self, 
                 depth_scaling_factor,
                 width_scaling_factor, 
                 depthwise=False):
        super().__init__()
        self.backbone = YOLOPAFPN(depth=depth_scaling_factor, width=width_scaling_factor, depthwise=depthwise)
    def forward(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        return fpn_outs