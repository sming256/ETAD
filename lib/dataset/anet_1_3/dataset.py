import os
import torch
import json
import math
import numpy as np
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope

from lib.utils.misc import iou_with_anchors, ioa_with_anchors, get_valid_mask, torchvision_pool_data
from lib.utils.sampler import ProposalSampler

"""
In anet, we don't use sliding window because many actions are quite long.
Thus, we rescale the video feature to fixed length.
"""


class VideoDataSet(torch.utils.data.Dataset):
    def __init__(self, mode="train", subset="training", cfg=None, logger=None):
        # MODE SETTINGS
        self.e2e = cfg.E2E_SETTING.mode
        self.mode = mode
        self.subset = subset
        self.anno_folder = "./lib/dataset/anet_1_3/data/"
        self.printer = logger.info if logger is not None else print

        # MODEL SETTINGS
        self.tscale = cfg.DATASET.tscale
        self.dscale = cfg.DATASET.dscale
        self.pos_thresh = cfg.LOSS.pos_thresh
        self.proposal_sampling_ratio = cfg.SAMPLING_RATIO.proposal
        self.proposal_sampling_strategy = cfg.SAMPLING_STRATEGY.proposal

        # FEATURE SETTING
        if self.e2e:
            init_default_scope("mmaction")
            if subset == "training":
                data_pipeline = Compose(cfg.TRAIN_PIPELINE)
            else:
                data_pipeline = Compose(cfg.TEST_PIPELINE)
            self.data_pipeline = data_pipeline
            self.video_path = cfg.VIDEO_PATH
            self.printer("{} pipeline: {}".format(subset, data_pipeline))
        else:
            self.feature_path = cfg.FEATURE.path
            self.transpose = getattr(cfg.FEATURE, "transpose", True)
            self.online_resize = cfg.FEATURE.online_resize

        self._get_dataset()
        self._get_anchors()
        self._get_mask()

    def _get_dataset(self):
        anno_database = json.load(open(self.anno_folder + "activity_net.v1-3.min.json"))
        anno_database = anno_database["database"]

        self.video_dict = {}
        for video_name, video_info in anno_database.items():
            if video_info["subset"] != self.subset:
                continue

            # in training, we filter some videos with wrong annotation, such as annotated with [0,0.01]
            # but in testing, we do not block any videos for fair comparison
            if video_info["subset"] == "training":
                gt_cnt = 0
                for gt in video_info["annotations"]:
                    tmp_start = max(min(1, gt["segment"][0] / video_info["duration"]), 0)
                    tmp_end = max(min(1, gt["segment"][1] / video_info["duration"]), 0)
                    if tmp_end - tmp_start > 0.01:  # if less than 0.01 * video length
                        gt_cnt += 1
                if gt_cnt == 0:
                    continue

            self.video_dict[video_name] = video_info

        self.video_list = list(self.video_dict.keys())
        self.printer("{} subset video numbers: {} ".format(self.subset, len(self.video_list)))

    def _get_gts(self, index, anchors_init):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_second = video_info["duration"]
        video_labels = video_info["annotations"]

        gt_bbox = []
        for gt in video_labels:
            tmp_start = max(min(1, gt["segment"][0] / video_second), 0)
            tmp_end = max(min(1, gt["segment"][1] / video_second), 0)
            # filter some dirty actions
            if tmp_end - tmp_start < 0.01 and self.subset == "train":
                continue
            else:
                gt_bbox.append([tmp_start, tmp_end])
        gt_bbox = np.array(gt_bbox)

        # gt start end
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        bboxes_len = 3.0 / self.tscale
        gt_start_bboxs = np.stack((gt_xmins - bboxes_len / 2, gt_xmins + bboxes_len / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - bboxes_len / 2, gt_xmaxs + bboxes_len / 2), axis=1)

        gt_start = ioa_with_anchors(gt_start_bboxs, self.temporal_anchor)
        gt_start = np.max(gt_start, axis=1)

        gt_end = ioa_with_anchors(gt_end_bboxs, self.temporal_anchor)  # [T, N]
        gt_end = np.max(gt_end, axis=1)

        # gt bbox
        ious = iou_with_anchors(gt_bbox, anchors_init.numpy())  # [K,N]
        gt_iou_index = np.argmax(ious, axis=1)  # [K]
        gt_bbox = gt_bbox[gt_iou_index]  # [K,2]

        # corresponding gt to tensor
        gt_start = torch.Tensor(gt_start)
        gt_end = torch.Tensor(gt_end)
        gt_bbox = torch.Tensor(gt_bbox)  # [K,2]
        return (gt_start, gt_end, gt_bbox)

    def _get_video_data(self, index):
        video_info = {}
        video_name = self.video_list[index]
        video_info["video_name"] = video_name
        video_info["indices"] = self.anchor_xmin

        if self.e2e:
            video_data = self._get_video_frames(video_name)
        else:  # offline feature: v_xxxxxx.npy
            video_data = np.load(os.path.join(self.feature_path, "v_{}.npy".format(video_name)))
            video_data = torch.from_numpy(video_data).float()

            if self.transpose:
                video_data = video_data.transpose(1, 0)  # feature [T,C] -> [C,T]

            if self.online_resize:
                video_data = torchvision_pool_data(video_data, tscale=self.tscale)

        # get init anchors
        anchors_init = self._proposal_sampling()
        return video_info, video_data, anchors_init

    def _get_video_frames(self, video_name):
        results = dict(
            filename=os.path.join(self.video_path, f"v_{video_name}.mp4"),
            start_index=0,
            modality="RGB",
        )
        data = self.data_pipeline(results)
        imgs = data["imgs"]  # [N, 3, T, H, W]
        return imgs

    def _proposal_sampling(self):
        if self.mode == "train":
            if self.subset == "training":
                anchors_init = ProposalSampler(
                    self.anchors,
                    method=self.proposal_sampling_strategy,  # random, grid, block, fps, dpp
                    sampling_ratio=self.proposal_sampling_ratio,
                    valid_mask=self.valid_mask,
                    grid_mask=self.grid_mask,
                    tscale=self.tscale,
                    dscale=self.dscale,
                )
            elif self.subset == "validation":
                anchors_init = self.anchors[self.grid_mask.bool()]
        else:
            anchors_init = self.anchors[self.valid_mask.bool()]
        return anchors_init

    def _get_anchors(self):
        self.anchor_xmin = np.array([i / self.tscale for i in range(self.tscale)])
        self.anchor_xmax = np.array([i / self.tscale for i in range(1, self.tscale + 1)])
        self.temporal_anchor = np.stack([self.anchor_xmin, self.anchor_xmax], axis=1)

        map_anchor = []
        for idx in range(self.dscale):
            for jdx in range(self.tscale):
                if jdx + idx < self.tscale:
                    xmin = float(self.anchor_xmin[jdx])
                    xmax = float(self.anchor_xmax[jdx + idx])
                    map_anchor.append([xmin, xmax])
                else:
                    map_anchor.append([0, 0])
        self.anchors = torch.Tensor(map_anchor).reshape(self.dscale, self.tscale, -1)

    def _get_mask(self):
        # valid mask
        valid_mask = get_valid_mask(self.dscale, self.tscale)
        self.valid_mask = torch.Tensor(valid_mask)

        # grid mask for validation
        step = math.sqrt(1 / self.proposal_sampling_ratio)
        mask = np.zeros((self.dscale, self.tscale))
        for idx in np.arange(1, self.dscale, step).round().astype(int):
            for jdx in np.arange(1, self.tscale, step).round().astype(int):
                if jdx + idx < self.tscale:
                    mask[idx, jdx] = 1
        self.grid_mask = torch.Tensor(mask)

    def __getitem__(self, index):
        video_info, video_data, anchors_init = self._get_video_data(index)
        # anchors_init[K,2] 0~1

        if self.mode == "train":
            gts = self._get_gts(index, anchors_init)
            return video_info, video_data, anchors_init, gts
        elif self.mode == "infer":
            return video_info, video_data, anchors_init

    def __len__(self):
        return len(self.video_list)
