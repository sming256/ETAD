import os
import random
import torch
import torchvision
import pandas as pd
import numpy as np


def set_seed(seed):
    """Set random seed for pytorch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_name(config, run_id):
    # update config name
    if run_id == -1:
        config.EXP_NAME += "/default"
    else:
        config.EXP_NAME += "/id{}".format(run_id)
    return config


def create_folder(cfg):
    output_dir = "./exps/%s/" % (cfg.EXP_NAME)
    if not os.path.exists(output_dir):
        os.system("mkdir -p ./exps/%s/" % (cfg.EXP_NAME))


def create_infer_folder(cfg):
    output_path = "./exps/%s/output/" % (cfg.EXP_NAME)
    if not os.path.exists(output_path):
        os.system("mkdir -p %s" % (output_path))


def save_config(cfg, exp_name):
    os.system("cp {} ./exps/{}/{}".format(cfg, exp_name, cfg.split("/")[-1]))


def iou_with_anchors(gt_boxes, anchors):
    """Compute IoU between gt_boxes and anchors.
    gt_boxes: np.array shape [N, 2]
    anchors:  np.array shape [D*T, 2]
    """

    N = gt_boxes.shape[0]
    M = anchors.shape[0]

    gt_areas = (gt_boxes[:, 1] - gt_boxes[:, 0]).reshape(1, N)
    anchors_areas = (anchors[:, 1] - anchors[:, 0]).reshape(M, 1)

    boxes = anchors.reshape(M, 1, 2).repeat(N, axis=1)
    query_boxes = gt_boxes.reshape(1, N, 2).repeat(M, axis=0)

    inter_max = np.minimum(boxes[:, :, 1], query_boxes[:, :, 1])
    inter_min = np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
    inter = np.maximum(inter_max - inter_min, 0.0)

    scores = inter / (anchors_areas + gt_areas - inter + 1e-6)  # shape [D*T, N]
    return scores


def ioa_with_anchors(gt_boxes, anchors):
    """Compute Intersection between gt_boxes and anchors.
    gt_boxes: np.array shape [N, 2]
    anchors:  np.array shape [T, 2]
    """

    N = gt_boxes.shape[0]
    M = anchors.shape[0]

    anchors_areas = (anchors[:, 1] - anchors[:, 0]).reshape(M, 1)

    boxes = anchors.reshape(M, 1, 2).repeat(N, axis=1)
    query_boxes = gt_boxes.reshape(1, N, 2).repeat(M, axis=0)

    inter_max = np.minimum(boxes[:, :, 1], query_boxes[:, :, 1])
    inter_min = np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
    inter = np.maximum(inter_max - inter_min, 0.0)

    scores = inter / (anchors_areas + 1e-6)  # shape [T, N]
    return scores


def boundary_choose(score_list):
    max_score = max(score_list)
    mask_high = score_list > max_score * 0.5
    score_list = list(score_list)
    score_middle = np.array([0.0] + score_list + [0.0])
    score_front = np.array([0.0, 0.0] + score_list)
    score_back = np.array(score_list + [0.0, 0.0])
    mask_peak = (score_middle > score_front) & (score_middle > score_back)
    mask_peak = mask_peak[1:-1]
    mask = (mask_high | mask_peak).astype("float32")
    return mask


def get_valid_mask(dscale, tscale):
    mask = np.zeros((dscale, tscale))
    for idx in range(dscale):
        for jdx in range(tscale):
            if jdx + idx < tscale:
                mask[idx, jdx] = 1
    return mask


def iou_with_anchors_batch_aligned(gt_boxes, anchors):
    """Compute IoU between gt_boxes and anchors.
    gt_boxes: shape [B, N, 2]
    anchors:  shape [B, N, 2]
    gt_boxes has been aligned with anchors
    """
    bs = gt_boxes.shape[0]
    N = gt_boxes.shape[1]

    gt_areas = (gt_boxes[:, :, 1] - gt_boxes[:, :, 0]).view(bs, N)
    anchors_areas = (anchors[:, :, 1] - anchors[:, :, 0]).view(bs, N)

    inter_max = torch.min(anchors[:, :, 1], gt_boxes[:, :, 1])
    inter_min = torch.max(anchors[:, :, 0], gt_boxes[:, :, 0])
    inter = (inter_max - inter_min).clamp(min=0)

    scores = inter / (anchors_areas + gt_areas - inter)  # [B,N]
    return scores


def reg_to_anchors(anchors, regs):
    """convert initial anchors with regressions

    Args:
        anchors ([tensor]): [B,K,2] (0~1)
        regs ([tensor]): [B*K,6]
    """
    regs = regs.view(anchors.shape[0], -1, regs.shape[-1]).detach()  # [B,K,6]

    xmins = anchors[:, :, 0]
    xmaxs = anchors[:, :, 1]
    xlens = xmaxs - xmins
    xcens = (xmins + xmaxs) * 0.5

    # refine anchor by start end
    xlens1 = xlens + xlens * (regs[:, :, 1] - regs[:, :, 0])
    xcens1 = xcens + xlens * (regs[:, :, 0] + regs[:, :, 1]) * 0.5
    xmins1 = xcens1 - xlens1 * 0.5
    xmaxs1 = xcens1 + xlens1 * 0.5

    # refine anchor by center width
    xcens2 = xcens + regs[:, :, 2] * xlens
    xlens2 = xlens * torch.exp(regs[:, :, 3])
    xmins2 = xcens2 - xlens2 * 0.5
    xmaxs2 = xcens2 + xlens2 * 0.5

    nxmin = (xmins1 + xmins2) * 0.5
    nxmax = (xmaxs1 + xmaxs2) * 0.5
    new_anchors = torch.stack((nxmin, nxmax), dim=2)
    return new_anchors


def torchvision_pool_data(feat, tscale=128):
    # input feat shape [C,T]
    pseudo_input = feat.unsqueeze(0).unsqueeze(3)  # [1,C,T,1]
    pseudo_bbox = torch.Tensor([[0, 0, 0, 1, feat.shape[1]]])
    # convert to half().double() to avoid non-deterministic behavior
    output = torchvision.ops.roi_align(
        pseudo_input.half().double(),
        pseudo_bbox.half().double(),
        output_size=(tscale, 1),
        aligned=True,
    ).to(
        pseudo_input.dtype
    )  # [1,C,tscale,1]
    output = output.squeeze(0).squeeze(-1)  # [C,tscale]
    return output


def soft_nms(df, iou_threshold, sigma, max_num=100):
    """
    df: proposals generated by network;
    alpha: alpha value of Gaussian decaying function;
    t1, t2: threshold for soft nms.
    """
    df = df.sort_values(by="score", ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < max_num:
        max_index = tscore.index(max(tscore))
        tmp_start = tstart[max_index]
        tmp_end = tend[max_index]
        tmp_score = tscore[max_index]
        rstart.append(tmp_start)
        rend.append(tmp_end)
        rscore.append(tmp_score)
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

        tstart = np.array(tstart)
        tend = np.array(tend)
        tscore = np.array(tscore)

        tt1 = np.maximum(tmp_start, tstart)
        tt2 = np.minimum(tmp_end, tend)
        intersection = tt2 - tt1
        duration = tend - tstart
        tmp_width = tmp_end - tmp_start
        iou = intersection / (tmp_width + duration - intersection).astype(float)

        idxs = np.where(iou > iou_threshold)[0]
        tscore[idxs] = tscore[idxs] * np.exp(-np.square(iou[idxs]) / sigma)

        tstart = list(tstart)
        tend = list(tend)
        tscore = list(tscore)

    newDf = pd.DataFrame()
    newDf["score"] = rscore
    newDf["xmin"] = rstart
    newDf["xmax"] = rend
    return newDf
