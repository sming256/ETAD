import torch
import os
import tqdm
import pickle
from lib.utils.misc import reg_to_anchors


def inference(model, data_loader, logger, cfg):
    output_path = "./exps/{}/output/".format(cfg.EXP_NAME)

    for video_info, video_data, anchors_init in tqdm.tqdm(data_loader):
        batch_size = video_data.shape[0]
        video_data = video_data.cuda()
        anchors_init = anchors_init.cuda()

        with torch.no_grad():
            (tem_out, stage_out) = model(video_data, anchors_init=anchors_init)

        # get anchors and ious
        anchors = torch.stack([reg_to_anchors(out[0], out[2]) for out in stage_out], dim=0).mean(dim=0)
        ious = torch.stack([out[1] for out in stage_out], dim=0).mean(dim=0)
        ious = ious.view(batch_size, -1, ious.shape[1])

        for jdx in range(batch_size):
            # get snippet info
            video_name = video_info["video_name"][jdx]
            video_snippets = video_info["indices"][jdx].numpy()
            start = video_snippets[0]
            end = video_snippets[-1]

            # detach result
            pred_anchors = anchors[jdx].cpu().detach().numpy()
            pred_start = tem_out[jdx, 0, :].cpu().detach().numpy()
            pred_end = tem_out[jdx, 1, :].cpu().detach().numpy()
            pred_iou = ious[jdx].cpu().detach().numpy()

            result = [video_snippets, pred_anchors, pred_start, pred_end, pred_iou]

            # save result
            if cfg.DATASET.name in ["anet_1_3", "hacs"]:
                file_path = os.path.join(output_path, "{}.pkl".format(video_name))
            elif cfg.DATASET.name == "thumos_14":
                output_folder = os.path.join(output_path, video_name)
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                file_path = os.path.join(output_folder, "{}_{}.pkl".format(start, end))

            with open(file_path, "wb") as outfile:
                pickle.dump(result, outfile, pickle.HIGHEST_PROTOCOL)
