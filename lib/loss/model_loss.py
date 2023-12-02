import torch
import torch.nn.functional as F
from ..utils.misc import iou_with_anchors_batch_aligned


class TAL_loss(object):
    def __init__(self, cfg):
        super(TAL_loss, self).__init__()
        self.cfg = cfg

    def __call__(self, pred, video_gt):
        # unpack gt and prediction
        gt_start, gt_end, gt_bbox = video_gt
        tem_out, stage_out = pred

        # ---------- temporal evaluation loss ----------
        loss_start = bce_balance_loss(gt_start, tem_out[:, 0, :])
        loss_end = bce_balance_loss(gt_end, tem_out[:, 1, :])
        loss_tem = loss_start + loss_end

        # ---------- proposal evaluation loss ----------
        loss_pem_stage = []
        for i in range(len(stage_out)):
            loss_pem_stage.append(
                self._get_stage_loss(
                    gt_bbox,
                    gt_start,
                    gt_end,
                    stage_out[i],
                    reg_thresh=self.cfg.MODEL.stage[i],
                )
            )

        # ------------ total cost -----------
        cost = loss_tem + torch.stack(loss_pem_stage, dim=0).sum()

        loss_dict = {}
        loss_dict["cost"] = cost
        loss_dict["tem"] = loss_tem
        for i in range(len(stage_out)):
            loss_dict["stage{}".format(i + 1)] = loss_pem_stage[i]
        return cost, loss_dict

    def _get_stage_loss(self, gt_bbox, gt_start, gt_end, stage_out, reg_thresh=0.6):
        # gt_bbox [B,K,2] anchors_init [B,K,2] iou_out [B*K,2] reg [B*K,-1]
        anchors, iou, reg = stage_out
        gt_iou = iou_with_anchors_batch_aligned(gt_bbox, anchors).view(-1)  # [B*K]
        gt_delta_s, gt_delta_e, gt_delta_c, gt_delta_w = self._get_gt_regs(gt_bbox, anchors)  # [B*K,-1]
        gt_cls_s, gt_cls_e = self._get_gt_cls(gt_start, gt_end, anchors)

        # iou
        loss_pem_iou_cls = bce_sample_loss(iou[:, 0], gt_iou, pos_thresh=0.9)
        loss_pem_iou_reg = l2_sample_loss(iou[:, 1], gt_iou, high_thresh=0.7, low_thresh=0.3)

        loss_pem_iou_cls *= self.cfg.LOSS.coef_pem_cls
        loss_pem_iou_reg *= self.cfg.LOSS.coef_pem_reg
        loss_pem_iou = loss_pem_iou_reg + loss_pem_iou_cls

        # boundary regression
        loss_pem_bnd_start = smoothL1_regress_loss(reg[:, 0], gt_iou, gt_delta_s, thresh=reg_thresh)
        loss_pem_bnd_end = smoothL1_regress_loss(reg[:, 1], gt_iou, gt_delta_e, thresh=reg_thresh)

        loss_pem_bnd_center = smoothL1_regress_loss(reg[:, 2], gt_iou, gt_delta_c, thresh=reg_thresh)
        loss_pem_bnd_width = smoothL1_regress_loss(reg[:, 3], gt_iou, gt_delta_w, thresh=reg_thresh)

        loss_pem_bnd_reg_cw = self.cfg.LOSS.coef_pem_bnd * (loss_pem_bnd_center + loss_pem_bnd_width)
        loss_pem_bnd_reg_se = self.cfg.LOSS.coef_pem_bnd * (loss_pem_bnd_start + loss_pem_bnd_end)

        # boundary classification
        loss_pem_cls_start = bce_sample_loss(reg[:, 4], gt_cls_s, pos_thresh=0.5)
        loss_pem_cls_end = bce_sample_loss(reg[:, 5], gt_cls_e, pos_thresh=0.5)

        loss_pem_bnd_cly = 0.5 * (loss_pem_cls_start + loss_pem_cls_end)

        # total loss
        loss_pem_stage = loss_pem_iou + loss_pem_bnd_reg_cw + loss_pem_bnd_reg_se + loss_pem_bnd_cly
        return loss_pem_stage

    def _get_gt_regs(self, gt, anchors):
        # gt_bbox [B,K,2] anchors_init[B,K,2]
        anchor_len = torch.clamp(anchors[:, :, 1] - anchors[:, :, 0], min=1e-6)
        gt_len = torch.clamp(gt[:, :, 1] - gt[:, :, 0], min=1e-6)
        delta_s = (gt[:, :, 0] - anchors[:, :, 0]) / anchor_len
        delta_e = (gt[:, :, 1] - anchors[:, :, 1]) / anchor_len
        delta_c = (delta_s + delta_e) * 0.5
        delta_w = torch.log(gt_len / anchor_len + 1e-6)
        return delta_s.view(-1), delta_e.view(-1), delta_c.view(-1), delta_w.view(-1)

    def _get_gt_cls(self, gt_start, gt_end, anchors):
        # gt_start [B,200] anchors_init[B,K,2]
        anchors_start = anchors[:, :, 0] * self.cfg.DATASET.tscale
        anchors_end = anchors[:, :, 1] * self.cfg.DATASET.tscale
        anchors_start = anchors_start.long().clamp(min=0, max=gt_start.shape[1] - 1)
        anchors_end = anchors_end.long().clamp(min=0, max=gt_end.shape[1] - 1)
        gt_cls_s = torch.gather(gt_start, 1, anchors_start)
        gt_cls_e = torch.gather(gt_end, 1, anchors_end)
        return gt_cls_s.view(-1), gt_cls_e.view(-1)


def bce_balance_loss(gt, pred, pos_thresh=0.5):
    """Balanced Cross Entropy Loss"""
    gt = gt.view(-1).cuda()
    pred = pred.contiguous().view(-1)

    pmask = (gt > pos_thresh).float().cuda()
    nmask = (gt <= pos_thresh).float().cuda()

    num_pos = torch.sum(pmask)
    num_neg = torch.sum(nmask)

    if num_pos == 0:  # not have positive sample
        loss = nmask * torch.log(1.0 - pred + 1e-6)
        loss = -torch.sum(loss) / num_neg
        return loss

    if num_neg == 0:  # not have negative sample
        loss = pmask * torch.log(pred + 1e-6)
        loss = -torch.sum(loss) / num_pos
        return loss

    coef_pos = 0.5 * (num_pos + num_neg) / num_pos
    coef_neg = 0.5 * (num_pos + num_neg) / num_neg

    loss = coef_pos * pmask * torch.log(pred + 1e-6) + coef_neg * nmask * torch.log(1.0 - pred + 1e-6)
    loss = -torch.mean(loss)
    return loss


def bce_sample_loss(output, gt_iou, pos_thresh=0.9):
    gt_iou = gt_iou.cuda()

    pmask = (gt_iou > pos_thresh).float()
    nmask = (gt_iou <= pos_thresh).float()

    num_pos = torch.sum(pmask)
    num_neg = torch.sum(nmask)

    if num_pos == 0:  # in case of nan
        loss = -torch.mean(torch.log(1.0 - output + 1e-6))
        return loss

    r_l = num_pos / num_neg
    nmask_sample = torch.rand(nmask.shape).cuda()
    nmask_sample = nmask_sample * nmask
    nmask_sample = (nmask_sample > (1 - r_l)).float()

    loss_pos = pmask * torch.log(output + 1e-6)
    loss_neg = nmask_sample * torch.log(1.0 - output + 1e-6)
    loss = -torch.sum(loss_pos + loss_neg) / (num_pos + torch.sum(nmask_sample))
    return loss


def l2_sample_loss(output, gt_iou, high_thresh=0.7, low_thresh=0.3):
    gt_iou = gt_iou.cuda()

    u_hmask = (gt_iou > high_thresh).float()
    u_mmask = ((gt_iou <= high_thresh) & (gt_iou > low_thresh)).float()
    u_lmask = (gt_iou <= low_thresh).float()

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    if num_h == 0:  # in case of nan
        loss = F.mse_loss(output, gt_iou, reduction="none")
        loss = torch.mean(loss)
        return loss

    r_m = num_h / num_m
    u_smmask = torch.rand(u_hmask.shape).cuda()
    u_smmask = u_smmask * u_mmask
    u_smmask = (u_smmask > (1 - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.rand(u_hmask.shape).cuda()
    u_slmask = u_slmask * u_lmask
    u_slmask = (u_slmask > (1 - r_l)).float()

    mask = u_hmask + u_smmask + u_slmask
    loss = F.mse_loss(output, gt_iou, reduction="none")
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def smoothL1_regress_loss(output, gt_iou, gt_reg, thresh=0.7):
    mask = (gt_iou > thresh).float().cuda()

    if torch.sum(mask) == 0:  # not have positive sample
        return torch.Tensor([0]).sum().cuda()

    loss = F.smooth_l1_loss(output, gt_reg, reduction="none")
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss
