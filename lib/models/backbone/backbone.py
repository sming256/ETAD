import torch
from mmengine.registry import MODELS
from mmengine.registry import init_default_scope


class backbone_model(torch.nn.Module):
    def __init__(self, cfg=None, logger=None):
        super(backbone_model, self).__init__()

        # build model
        model_config = cfg.E2E_SETTING.model
        init_default_scope("mmaction")
        logger.info("model config: {}".format(model_config))
        self.model = MODELS.build(model_config)
        self.model_type = model_config.type

    def forward(self, frames, train=True):
        # frames [B, T, 3, 16, 224, 224]

        # data preprocessing: normalize mean and std
        frames, _ = self.model.data_preprocessor.preprocess(
            [t for t in frames],  # need list input
            data_samples=None,
            training=False,  # for blending, which is not used
        )

        self.model.train() if train else self.model.eval()
        with torch.set_grad_enabled(train):
            video_feat = self.model_forward(frames)  # [B,C,T]

        return video_feat

    def model_forward(self, frames):
        batches, num_segs = frames.shape[0:2]  #  [B, T, C, L, H, W]

        if self.model_type == "Recognizer2D":
            raise NotImplementedError
        elif self.model_type == "Recognizer3D":
            frames = frames.flatten(0, 1)  # [B*T,C,L,H,W]
            video_feat = self.model.backbone(frames)  # [B*T,C,t,h,w]
            video_feat = video_feat.mean(dim=[2, 3, 4])
            video_feat = video_feat.view(batches, num_segs, -1).permute(0, 2, 1)  # [B,C,T]
        else:
            raise NotImplementedError

        return video_feat
