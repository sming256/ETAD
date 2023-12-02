import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import torch
import argparse
from mmengine.config import Config, DictAction
from lib.dataset import build_dataset
from lib.models.build_model import TAL_model
from lib.loss.model_loss import TAL_loss
from lib.core.trainer import train_one_epoch
from lib.utils.misc import set_seed, update_name, create_folder, save_config
from lib.utils.logger import setup_logger


def train(cfg, logger):
    # build dataset and dataloader
    logger.info("dataset: {}".format(cfg.DATASET.name))
    train_dataset = build_dataset(name=cfg.DATASET.name, mode="train", subset="training", cfg=cfg, logger=logger)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.SOLVER.batch_size,
        num_workers=cfg.SOLVER.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # build model and loss function
    model = TAL_model(cfg, logger=logger)
    criterion = TAL_loss(cfg)

    # build optimizer
    optimizer = model.get_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.step_size, gamma=cfg.SOLVER.gamma)

    # if need resume, load the checkpoint
    if cfg.resume != "":
        logger.info("Resume training from: {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location="cpu")
        logger.info("Checkpoint is epoch {}".format(checkpoint["epoch"]))
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    else:
        start_epoch = 0

    # DataParallel
    logger.info("Using DP, batch size is {}, GPU num is {}".format(cfg.SOLVER.batch_size, cfg.num_gpus))
    model = torch.nn.DataParallel(model, device_ids=list(range(cfg.num_gpus))).cuda()

    # training
    logger.info("Start training")
    for epoch in range(start_epoch, cfg.SOLVER.epoch):
        cfg.epoch = epoch
        train_one_epoch(model, criterion, train_loader, logger, cfg, optimizer, scheduler)
    logger.info("Train over.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETAD")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("num_gpus", type=int)
    parser.add_argument("--resume", type=str, default="", help="resume from checkpoint")
    parser.add_argument("--id", type=int, default=-1, help="for multi-runs, default=-1")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()

    # load settings
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.num_gpus = args.num_gpus
    cfg.resume = args.resume
    cfg = update_name(cfg, args.id)
    set_seed(2023)

    # create work folder
    create_folder(cfg)

    # setup logger and save config
    logger = setup_logger("Train", "./exps/%s/" % (cfg.EXP_NAME))
    logger.info("Using torch version: {}, CUDA version: {}".format(torch.__version__, torch.version.cuda))
    logger.info("Using {} GPUs".format(cfg.num_gpus))
    logger.info(cfg)
    save_config(args.config, cfg.EXP_NAME)

    # start training
    train(cfg, logger)
