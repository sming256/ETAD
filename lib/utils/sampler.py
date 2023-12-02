import random
import math
import torch
import numpy as np


def ProposalSampler(anchors, valid_mask, grid_mask, method="random", sampling_ratio=0.06, tscale=128, dscale=128):
    """This is the function to sample a subset of given proposals."""

    sample_num = int(valid_mask.sum().int() * sampling_ratio)

    if method == "random":  # random select
        indices = random.sample(range(valid_mask.sum().int()), sample_num)
        indices = torch.Tensor(indices).long()
        indices = torch.nonzero(valid_mask)[indices]
        select_mask = torch.zeros_like(valid_mask)
        select_mask[indices[:, 0], indices[:, 1]] = 1
        anchors_init = anchors[select_mask.bool()]

    elif method == "grid":  # grid select
        anchors_init = anchors[grid_mask.bool()]

    elif method == "block":  # block select
        block_w = int(math.sqrt(sample_num))
        x = random.randint(0, dscale - 2 * block_w)
        y = random.randint(0, tscale - 2 * block_w - x)
        mask = np.zeros((dscale, tscale))
        for idx in range(x, x + block_w):
            for jdx in range(y, y + block_w):
                if jdx + idx < tscale:
                    mask[idx, jdx] = 1
        block_mask = torch.Tensor(mask)
        anchors_init = anchors[block_mask.bool()]

    # shuffle the anchors to avoid over fitting
    idx = torch.randperm(anchors_init.shape[0])
    anchors_init = anchors_init[idx, :]
    return anchors_init
