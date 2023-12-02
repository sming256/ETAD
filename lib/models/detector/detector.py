import sys
import os

sys.path.append(os.path.dirname(__file__) + "../../")

import copy
import torch.nn as nn
from lib.models.detector.feature import feature_module
from lib.models.detector.pem import pem_module
from lib.utils.misc import reg_to_anchors


class detector_model(nn.Module):
    """This is an efficient detector, mainly contains three modules:
    1.Feature Module: enhance the feature sequence
    2.Temporal Evaluation Module: predict boundary probabilities
    3.Proposal Evaluation Module: predict the IoU, start/end of proposals
    """

    def __init__(self, cfg):
        super(detector_model, self).__init__()
        self.stage_num = len(cfg.MODEL.stage)

        # temporal enhancement module
        self.temporal_enhance = feature_module(in_dim=cfg.MODEL.in_channels)

        # temporal evaluation
        self.tem = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 2, kernel_size=1),
            nn.Sigmoid(),
        )

        # proposal evaluation
        pem_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
        )
        pem = pem_module(
            in_dim=256,
            tscale=cfg.DATASET.tscale,
            roi_size=cfg.MODEL.roi_size,
        )
        self.stage_conv = nn.ModuleList([copy.deepcopy(pem_conv) for _ in range(self.stage_num)])
        self.stage_pem = nn.ModuleList([copy.deepcopy(pem) for _ in range(self.stage_num)])
        self.reset_params()

    def forward(self, x, anchors):
        #  temporal enhancement module
        x = self.temporal_enhance(x)  # [B,C,T]

        # temporal evaluation
        tem_out = self.tem(x)  # [B,2,T]

        # proposal evaluation
        stage_out = []
        for i in range(self.stage_num):
            x_stage = x + self.stage_conv[i](x)
            iou, reg = self.stage_pem[i](x_stage, anchors)
            stage_out.append((anchors, iou, reg))  # iou [N,2], reg [N,2]
            anchors = reg_to_anchors(anchors, reg)

        return tem_out, stage_out

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
