import os
import sys
import numpy as np
import pandas as pd
import json
import time
import multiprocessing as mp
import pickle

path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

from lib.utils.misc import boundary_choose, soft_nms


def _gen_detection_video(video_list, video_dict, cuhk_data, cfg, num_prop=100, topk=2):
    tscale = cfg.DATASET.tscale
    output_path = "./exps/{}/output/".format(cfg.EXP_NAME)

    cols = ["xmin", "xmax", "score"]
    cuhk_data_score = cuhk_data["results"]
    cuhk_data_action = cuhk_data["class"]

    for video_name in video_list:
        file_path = os.path.join(output_path, "{}.pkl".format(video_name))

        with open(file_path, "rb") as infile:
            result = pickle.load(infile)

        [_, pred_anchors, pred_s, pred_e, pred_iou] = result

        # anchors [N,2]  pred_iou[N,6]
        start_mask = boundary_choose(pred_s)  # [100]
        start_mask[0] = 1.0
        end_mask = boundary_choose(pred_e)
        end_mask[-1] = 1.0

        ranchors = pred_anchors * tscale
        start_idx = np.clip(ranchors[:, 0].astype(int), 0, tscale - 1)
        end_idx = np.clip(ranchors[:, 1].astype(int), 0, tscale - 1)
        idx = np.where((start_mask[start_idx] == 1) & (end_mask[end_idx] == 1))

        xmins = pred_anchors[:, 0][idx]
        xmaxs = pred_anchors[:, 1][idx]

        pred_iou = pred_iou[:, 0] * pred_iou[:, 1]
        conf_score = pred_iou[idx]

        score_vector_list = np.stack((xmins, xmaxs, conf_score), axis=1)
        df = pd.DataFrame(score_vector_list.astype(float), columns=cols)

        if len(df) > 1:
            df = soft_nms(
                df,
                iou_threshold=cfg.DETECTION_POST.iou_threshold,
                sigma=cfg.DETECTION_POST.sigma,
            )
        df = df.sort_values(by="score", ascending=False)

        # sort video classification
        cuhk_score = np.array(cuhk_data_score[video_name])
        cuhk_data_action = np.array(cuhk_data_action)
        cuhk_classes = cuhk_data_action[np.argsort(-cuhk_score)]
        cuhk_score = cuhk_score[np.argsort(-cuhk_score)]

        video_duration = video_dict[video_name]["duration"]
        proposal_list = []
        for j in range(min(num_prop, len(df))):
            for k in range(topk):
                tmp_proposal = {}
                tmp_proposal["label"] = cuhk_classes[k]
                tmp_proposal["score"] = df.score.values[j] * cuhk_score[k]
                tmp_proposal["segment"] = [
                    max(0, df.xmin.values[j]) * video_duration,
                    min(1, df.xmax.values[j]) * video_duration,
                ]
                proposal_list.append(tmp_proposal)
        result_dict[video_name] = proposal_list


def gen_detection_multicore(cfg, subset="validation"):
    # get video list
    anno_database = json.load(open("./lib/dataset/anet_1_3/data/activity_net.v1-3.min.json"))["database"]
    anno_database = anno_database

    video_dict = {}
    for video_name, video_info in anno_database.items():
        if video_info["subset"] != subset:
            continue
        video_dict[video_name] = video_info
    video_list = list(video_dict.keys())

    # detection_result
    cuhk_data = json.load(open("./lib/dataset/anet_1_3/data/cuhk_val_simp_7.json"))

    global result_dict
    result_dict = mp.Manager().dict()

    # multi processing
    pp_num = 16
    num_videos_per_thread = len(video_list) / pp_num
    processes = []
    for tid in range(pp_num):
        num_start = int(tid * num_videos_per_thread)
        num_end = min(len(video_list), int((tid + 1) * num_videos_per_thread))
        p = mp.Process(
            target=_gen_detection_video,
            args=(video_list[num_start:num_end], video_dict, cuhk_data, cfg),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # write file
    result_dict = dict(result_dict)
    output_dict = {
        "version": "ActivityNet 1.3",
        "results": result_dict,
        "external_data": {},
    }

    with open(cfg.result_path, "w") as out:
        json.dump(output_dict, out)


def detection_post(cfg, logger):
    cfg.output_path = "./exps/%s/output/" % (cfg.EXP_NAME)
    cfg.result_path = "./exps/%s/result_detection.json" % (cfg.EXP_NAME)

    # post processing
    t1 = time.time()
    logger.info("Detection task post processing start")
    gen_detection_multicore(cfg)
    t2 = time.time()
    logger.info("Detection task post processing finished, time=%.1fmins" % ((t2 - t1) / 60))

    # evaluation
    from lib.eval.eval_detection import ANETdetection

    logger.info("Evaluating...")
    tious = np.linspace(0.5, 0.95, 10)
    anet_detection = ANETdetection(
        ground_truth_filename="./lib/dataset/anet_1_3/data/activity_net.v1-3.min.json",
        prediction_filename=cfg.result_path,
        tiou_thresholds=tious,
        subset="validation",
        blocked_videos="lib/dataset/anet_1_3/data/blocked.json",  # same validation results compared to BMN, etc.
        pp_num=16,
        verbose=False,
    )

    mAPs, average_mAP = anet_detection.evaluate()

    logger.info("Average-mAP: {}".format(average_mAP))
    for tiou, mAP in zip(tious, mAPs):
        if tiou in [0.5, 0.75, 0.95]:
            logger.info("mAP at tIoU {:.2f} is {:.2f}%".format(tiou, mAP * 100))

    # save to file
    cfg.eval_path = "./exps/%s/results.txt" % (cfg.EXP_NAME)
    f2 = open(cfg.eval_path, "a")
    f2.write("Average-mAP: {}\n".format(average_mAP))
    for tiou, mAP in zip(tious, mAPs):
        f2.write("mAP at tIoU {:.2f} is {:.2f}%\n".format(tiou, mAP * 100))
    f2.close()
