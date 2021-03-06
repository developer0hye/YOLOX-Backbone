#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        input_tensor_channels=3,
        depth=1.0,
        width=1.0,
        in_features=("C3", "C4", "C5"),
        out_features=["P3", "P4", "P5"],
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(input_tensor_channels, depth, width, depthwise=depthwise, act=act)
        self.scaling_factor = {"depth": depth, "width": width}
        
        self.in_features = in_features
        self.out_features = out_features

        self.in_channels = in_channels
        self.out_channels = {}
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.out_channels["P3"] = int(in_channels[0] * width)

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.out_channels["P4"] = int(in_channels[1] * width)

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.out_channels["P5"] = int(in_channels[2] * width)
        
        self.p3_exists = "P3" in self.out_features
        self.p4_exists = "P4" in self.out_features
        self.p5_exists = "P5" in self.out_features

    def forward(self, input):
        """
        Args:
            inputs: input images.
        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        
        if self.p3_exists or self.p4_exists or self.p5_exists:                    
            x2 = out_features["C3"]
            x1 = out_features["C4"]
            x0 = out_features["C5"]

            fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
            f_out0 = self.upsample(fpn_out0)  # 512/16
            f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
            f_out0 = self.C3_p4(f_out0)  # 1024->512/16

            fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
            f_out1 = self.upsample(fpn_out1)  # 256/8
            f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
            pan_out2 = self.C3_p3(f_out1)  # 512->256/8
            out_features["P3"] = pan_out2
            
            if self.p4_exists or self.p5_exists:
                p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
                p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
                pan_out1 = self.C3_n3(p_out1)  # 512->512/16
                out_features["P4"] = pan_out1
            
            if self.p5_exists:
                p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
                p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
                pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
                out_features["P5"] = pan_out0
            
        return {k:v for k, v in out_features.items() if k in self.out_features}
            