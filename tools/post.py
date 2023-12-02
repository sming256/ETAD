import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
from mmengine.config import Config, DictAction
from lib.dataset import build_post_processing
from lib.utils.misc import set_seed, update_name
from lib.utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETAD")
    parser.add_argument("config", metavar="FILE", type=str)  # path to config file
    parser.add_argument("--id", type=int, default=-1, help="for multi-runs, default=-1")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()

    # load settings
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = update_name(cfg, args.id)
    set_seed(2023)

    # setup logger
    logger = setup_logger("Post", "./exps/%s/" % (cfg.EXP_NAME))
    logger.info(cfg)

    # detection post processing according dataset    logger.info("dataset: {}".format(cfg.DATASET.name))
    logger.info("dataset: {}".format(cfg.DATASET.name))
    build_post_processing(name=cfg.DATASET.name, cfg=cfg, logger=logger)
