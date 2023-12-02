import time, copy
import os
import torch
import datetime
import pickle
from ..utils.metric_logger import MetricLogger


def train_one_epoch(model, criterion, data_loader, logger, cfg, optimizer=None, scheduler=None):
    model.train()

    meters = MetricLogger(delimiter="  ")
    end = time.time()

    max_iteration = len(data_loader)
    max_epoch = cfg.SOLVER.epoch
    last_epoch_iteration = (max_epoch - cfg.epoch - 1) * max_iteration

    for idx, (video_info, video_data, anchors_init, video_gt) in enumerate(data_loader):
        video_data = video_data.cuda()
        anchors_init = anchors_init.cuda()

        video_gt = [_gt.cuda() for _gt in video_gt]

        if not cfg.E2E_SETTING.mode:
            pred = model(video_data, anchors_init=anchors_init)
            cost, loss_dict = criterion(pred, video_gt)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        else:
            # stage 1: sequentially forward the backbone
            video_feat = model(video_data, stage=1)

            # stage 2: forward and backward the detector
            video_feat.requires_grad = True
            video_feat.retain_grad()
            det_pred = model(video_feat, anchors_init=anchors_init, stage=2)
            cost, loss_dict = criterion(det_pred, video_gt)

            # backward the detector
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # stage 3: sequentially backward the backbone with sampled data
            if cfg.SAMPLING_RATIO.snippet > 0:
                # copy the feature's gradient
                feat_grad = copy.deepcopy(video_feat.grad.detach())  # [B,C,T]

                # sample snippets and sequentially backward
                optimizer.zero_grad()
                model(video_data, feat_grad=feat_grad, stage=3)
                optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        meters.update(time=batch_time)
        meters.update(**loss_dict)

        eta_seconds = meters.time.avg * (max_iteration - idx - 1 + last_epoch_iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if ((idx % cfg.LOSS.log_interval == 0) and idx != 0) or (idx == max_iteration - 1):
            logger.info(
                meters.delimiter.join(
                    [
                        "{mode}: [E{epoch}/{max_epoch}]",
                        "iter: {iteration}/{max_iteration}",
                        "eta: {eta}",
                        "{meters}",
                        "max_mem: {memory:.2f}GB",
                    ]
                ).format(
                    mode="Train",
                    eta=eta_string,
                    epoch=cfg.epoch,
                    max_epoch=max_epoch - 1,
                    iteration=idx,
                    max_iteration=max_iteration - 1,
                    meters=str(meters),
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

    scheduler.step()
    save_checkpoint(model, cfg.epoch, cfg, scheduler, optimizer)


def save_checkpoint(model, epoch, cfg, scheduler, optimizer):
    exp_name = cfg.EXP_NAME

    state = {
        "epoch": epoch,
        "state_dict": model.module.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpoint_dir = "./exps/%s/checkpoint/" % (exp_name)

    if not os.path.exists(checkpoint_dir):
        os.system("mkdir -p %s" % (checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, "epoch_%d.pth.tar" % epoch)
    torch.save(state, checkpoint_path)
