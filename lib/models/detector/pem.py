import torch
import torch.nn as nn
import torchvision


class pem_module(nn.Module):
    def __init__(self, in_dim=256, mid_dim=512, fc_dim=128, roi_size=24, tscale=128, extend_ratio=0.5):
        super(pem_module, self).__init__()

        self.tscale = tscale
        self.roi_size = roi_size
        self.extend_ratio = extend_ratio

        self.reduce_dim_before = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # for extended feature
        self.reduce_dim_ext = nn.Sequential(
            nn.Linear(in_dim // 2 * self.roi_size, mid_dim),
            nn.ReLU(inplace=True),
        )
        self.iou_fc = nn.Sequential(
            nn.Linear(mid_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 2),
            nn.Sigmoid(),
        )
        self.cls_se_fc = nn.Sequential(
            nn.Linear(mid_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 2),
            nn.Sigmoid(),
        )
        self.reg_cw_fc = nn.Sequential(
            nn.Linear(mid_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 2),
        )

        # for start end feature
        self.reg_s_fc = nn.Sequential(
            nn.Linear(in_dim // 2 * self.roi_size // 4, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 1),
        )
        self.reg_e_fc = nn.Sequential(
            nn.Linear(in_dim // 2 * self.roi_size // 4, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x, anchors_init):
        x = self.reduce_dim_before(x)  # [B,C,T]

        # get proposal features
        ext_anchors = self._get_samples(anchors_init)  # anchors_init [B,K,2] 0~1
        iou_ft = self._get_proposal_features(x, ext_anchors)  # [B*K, C, res]

        # get features for start/end classification
        start_ft = iou_ft[:, :, : self.roi_size // 4]
        end_ft = iou_ft[:, :, -self.roi_size // 4 :]

        # reshape features
        start_ft = start_ft.reshape(start_ft.shape[0], -1)
        end_ft = end_ft.reshape(end_ft.shape[0], -1)
        iou_ft = iou_ft.reshape(iou_ft.shape[0], -1)
        # anchors[B*K, 3] anchor_ft[B*K, C*res]

        # for extended feature
        iou_ft = self.reduce_dim_ext(iou_ft)  # [B*K, 128]
        iou_out = self.iou_fc(iou_ft)
        cls_se = self.cls_se_fc(iou_ft)
        reg_cw = self.reg_cw_fc(iou_ft) / 2  # for smoother regression

        # for start end feature
        reg_s = self.reg_s_fc(start_ft)
        reg_e = self.reg_e_fc(end_ft)
        reg_out = torch.cat((reg_s, reg_e, reg_cw, cls_se), dim=1)
        return iou_out, reg_out

    def _get_samples(self, anchors_init):
        bs = anchors_init.shape[0]

        # convert from 0~1 to tscale
        anchors = anchors_init * self.tscale  # [B,K,2]

        # add batch idx
        bs_idxs = torch.arange(bs).view(bs, 1, 1).type_as(anchors_init)
        bs_idxs = bs_idxs.repeat(1, anchors.shape[1], 1).cuda()
        anchors = torch.cat((bs_idxs, anchors), dim=2)  # [B,K,3]
        anchors = anchors.view(-1, 3)  # [B*K,3]

        # get start anchors, end anchors
        anchors_len = anchors[:, 2] - anchors[:, 1]
        anchors = torch.stack(
            (
                anchors[:, 0],
                anchors[:, 1] - self.extend_ratio * anchors_len,
                anchors[:, 2] + self.extend_ratio * anchors_len,
            ),
            dim=1,
        )
        return anchors

    def _get_proposal_features(self, feature, proposals):
        # use torchvision align to do ROI align
        pseudo_input = feature.unsqueeze(3)  # [B,C,T,1]
        pseudo_bbox = torch.stack(
            (
                proposals[:, 0],
                torch.zeros_like(proposals[:, 0]),
                proposals[:, 1],
                torch.ones_like(proposals[:, 0]),
                proposals[:, 2],
            ),
            dim=1,
        )  # [B*K, 5]
        proposals_feat = torchvision.ops.roi_align(
            pseudo_input,
            pseudo_bbox,
            output_size=(self.roi_size, 1),
            aligned=False,
        )  # [B*K, C, roi_size, 1]
        return proposals_feat.squeeze(-1)  # [B*K, C, roi_size]
