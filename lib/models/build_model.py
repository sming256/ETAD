import torch
from .backbone.backbone import backbone_model
from .detector.detector import detector_model


class TAL_model(torch.nn.Module):
    def __init__(self, cfg, logger=None):
        super(TAL_model, self).__init__()

        self.E2E_mode = cfg.E2E_SETTING.mode

        if self.E2E_mode:
            self.feature_extractor = backbone_model(cfg, logger)
            self.chunk_size = cfg.E2E_SETTING.chunk_size
            self.sampling_ratio = cfg.SAMPLING_RATIO.snippet
            # self.sampling_strategy = cfg.SAMPLING_STRATEGY.snippet  # so far only support random sampling

        self.base_detector = detector_model(cfg)

    def forward(self, video_data, anchors_init=None, feat_grad=None, stage=0):
        if self.E2E_mode:  # end-to-end experiment
            if stage == 1:  # sequentially forward the backbone
                video_feat = self.forward_stage_1(video_data)
                return video_feat

            elif stage == 2:  # forward and backward the detector
                video_feat = video_data
                det_pred = self.base_detector(video_feat, anchors_init)
                return det_pred

            elif stage == 3:  # sequentially backward the backbone with sampled data
                self.forward_stage_3(video_data, feat_grad=feat_grad)

            elif stage == 0:  # this is for inference
                video_feat = self.forward_stage_1(video_data)
                det_pred = self.base_detector(video_feat, anchors_init)
                return det_pred

        else:  # feature-based experiment
            det_pred = self.base_detector(video_data, anchors_init)
            return det_pred

    def forward_stage_1(self, frames):
        # sequentially forward backbone
        chunk_num = frames.shape[1] // self.chunk_size  # frames [B,N,C,T,H,W]
        video_feat = []
        for mini_frames in torch.chunk(frames, chunk_num, dim=1):
            video_feat.append(self.feature_extractor(mini_frames, train=False))
        video_feat = torch.cat(video_feat, dim=2)

        # clean cache
        video_feat = video_feat.detach()
        torch.cuda.empty_cache()
        return video_feat

    def forward_stage_3(self, video_data, feat_grad):
        B, T, C, L, H, W = video_data.shape  # batch, snippet length, 3, clip length, h, w

        # sample the snippets
        chunk_num = int(T * self.sampling_ratio / self.chunk_size + 0.99)
        assert chunk_num > 0 and chunk_num * self.chunk_size <= T

        # random sampling
        noise = torch.rand(B, T, device=video_data.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)

        for chunk_idx in range(chunk_num):
            snippet_idx = ids_shuffle[:, chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size]

            video_data_chunk = torch.gather(
                video_data,
                dim=1,
                index=snippet_idx.view(B, self.chunk_size, 1, 1, 1, 1).repeat(1, 1, C, L, H, W),
            )
            feat_grad_chunk = torch.gather(
                feat_grad,
                dim=2,
                index=snippet_idx.view(B, 1, self.chunk_size).repeat(1, feat_grad.shape[1], 1),
            )

            video_feat_chunk = self.feature_extractor(video_data_chunk, train=True)
            assert video_feat_chunk.shape == feat_grad_chunk.shape

            # accumulate grad
            video_feat_chunk.backward(gradient=feat_grad_chunk)

    @staticmethod
    def get_optimizer(model, cfg):
        BACKBONE_weight = []
        FEAT_weight = []
        TEM_weight = []
        PEM_weight = []

        for name, p in model.named_parameters():
            if ("feature_extractor" in name) and ("detector" not in name):
                if ("bn" not in name) and (cfg.SAMPLING_RATIO.snippet > 0):
                    BACKBONE_weight.append(p)
            elif "feat" in name and "detector" in name:
                FEAT_weight.append(p)
            elif "tem" in name and "detector" in name:
                TEM_weight.append(p)
            elif "stage" in name and "detector" in name:
                PEM_weight.append(p)
            else:
                print("Attention! There are layers not loaded in optimizer :", name)

        optimizer = torch.optim.AdamW(
            [
                {"params": BACKBONE_weight, "lr": cfg.SOLVER.backbone_lr, "type": "backbone"},
                {"params": FEAT_weight, "lr": cfg.SOLVER.tal_lr, "type": "detector"},
                {"params": TEM_weight, "lr": cfg.SOLVER.tal_lr, "type": "detector"},
                {"params": PEM_weight, "lr": cfg.SOLVER.tal_lr, "type": "detector"},
            ],
            weight_decay=1e-4,
        )
        return optimizer
