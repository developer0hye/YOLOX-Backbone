#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import Darknet
from .network_blocks import BaseConv


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        input_tensor_channels=3,
        depth=53,
        in_features=["C3", "C4", "C5"],
        out_features=["P3", "P4", "P5"],
    ):
        super().__init__()

        self.backbone = Darknet(depth, in_channels=input_tensor_channels)
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = {"depth": 1.0, "width": 1.0}
        self.out_channels = {"P3": 128, "P4": 256, "P5": 512}

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.p3_exists = "P3" in self.out_features
        self.p4_exists = "P4" in self.out_features
        self.p5_exists = "P5" in self.out_features

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.
        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        if self.p3_exists or self.p4_exists or self.p5_exists:
            x2 = out_features["C3"]
            x1 = out_features["C4"]
            x0 = out_features["C5"]
            
            out_features["P5"] = x0
            
            if self.p3_exists or self.p4_exists:
                #  yolo branch 1
                x1_in = self.out1_cbl(x0)
                x1_in = self.upsample(x1_in)
                x1_in = torch.cat([x1_in, x1], 1)
                out_dark4 = self.out1(x1_in)
                out_features["P4"] = out_dark4

            if self.p3_exists:
                #  yolo branch 2
                x2_in = self.out2_cbl(out_dark4)
                x2_in = self.upsample(x2_in)
                x2_in = torch.cat([x2_in, x2], 1)
                out_dark3 = self.out2(x2_in)
                out_features["P3"] = out_dark3

        return {k:v for k, v in out_features.items() if k in self.out_features}