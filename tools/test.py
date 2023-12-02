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
from lib.core.inferer import inference
from lib.models.build_model import TAL_model
from lib.utils.misc import set_seed, update_name, create_infer_folder
from lib.utils.logger import setup_logger


def test(cfg, logger):
    # build dataset
    logger.info("dataset: {}".format(cfg.DATASET.name))
    test_dataset = build_dataset(name=cfg.DATASET.name, mode="infer", subset="validation", cfg=cfg, logger=logger)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.SOLVER.batch_size,
        num_workers=cfg.SOLVER.workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    # build model
    model = TAL_model(cfg, logger=logger)

    # load checkpoint
    if cfg.checkpoint != "None":  # load argparse epoch
        checkpoint_path = cfg.checkpoint
    elif "infer" in cfg.SOLVER:  # load config epoch
        checkpoint_path = "./exps/{}/checkpoint/epoch_{}.pth.tar".format(cfg.EXP_NAME, cfg.SOLVER.infer)
    else:
        raise ValueError("Please set the checkpoint path")

    logger.info("Loading checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    logger.info("Checkpoint is epoch {}".format(checkpoint["epoch"]))
    model.load_state_dict(checkpoint["state_dict"])

    # DataParallel
    model = torch.nn.DataParallel(model, device_ids=list(range(cfg.num_gpus))).cuda()
    model.eval()

    logger.info("Start inference")
    inference(model, test_loader, logger, cfg)
    logger.info("Inference over.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETAD")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("num_gpus", type=int)
    parser.add_argument("--checkpoint", type=str, default="None")
    parser.add_argument("--id", type=int, default=-1, help="for multi-runs, default=-1")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()

    # load settings
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.num_gpus = args.num_gpus
    cfg.checkpoint = args.checkpoint
    cfg = update_name(cfg, args.id)
    set_seed(2023)

    # create infer folder
    create_infer_folder(cfg)

    # setup logger
    logger = setup_logger("Infer", "./exps/%s/" % (cfg.EXP_NAME))
    logger.info("Using {} GPUs".format(cfg.num_gpus))
    logger.info(cfg)

    # start inference
    test(cfg, logger)
