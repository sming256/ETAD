EXP_NAME = "e2e_anet_tsp_snippet0.3_bs4_lr5e-7"

E2E_SETTING = dict(
    mode=True,
    chunk_size=4,  # snippet number of each chunk
    model=dict(
        type="Recognizer3D",
        backbone=dict(
            type="ResNet2Plus1d_TSP",
            layers=[3, 4, 6, 3],
            pretrained="pretrained/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth",
            frozen_stages=2,
            norm_eval=True,
        ),
        data_preprocessor=dict(
            type="ActionDataPreprocessor",
            mean=[110.2008, 100.63983, 95.99475],
            std=[58.14765, 56.46975, 55.332195],
            format_shape="NCTHW",
        ),
    ),
)

# DATASET SETTING
DATASET = dict(name="anet_1_3", tscale=128, dscale=128)
VIDEO_PATH = "data/anet/raw_data/Anet_videos_15fps_short256"
TRAIN_PIPELINE = [
    dict(type="DecordInit", num_threads=4),
    dict(type="SampleFrames", clip_len=16, num_clips=128, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(171, 128), keep_ratio=False),
    dict(type="RandomCrop", size=112),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="ImgAug", transforms="default"),
    dict(type="ColorJitter"),
    dict(type="FormatShape", input_format="NCTHW"),
]
TEST_PIPELINE = [
    dict(type="DecordInit", num_threads=4),
    dict(type="SampleFrames", clip_len=16, num_clips=128, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(171, 128), keep_ratio=False),
    dict(type="CenterCrop", crop_size=112),
    dict(type="FormatShape", input_format="NCTHW"),
]

# MODEL SETTINGS
MODEL = dict(in_channels=512, roi_size=24, stage=[0.7, 0.8, 0.9], extend_ratio=0.5)

# SAMPLING SETTINGS
SAMPLING_RATIO = dict(snippet=0.3, proposal=0.06)
SAMPLING_STRATEGY = dict(proposal="random", snippet="random")

# SOLVER SETTING
SOLVER = dict(
    tal_lr=5.0e-4,
    backbone_lr=5.0e-7,
    step_size=5,
    gamma=0.1,
    batch_size=4,
    workers=4,
    epoch=6,  # total epoch
    infer=5,  # infer epoch: 5 is the last epoch
)

# LOSS SETTING
LOSS = dict(
    log_interval=200,
    pos_thresh=0.9,
    coef_pem_cls=1,
    coef_pem_reg=5,
    coef_pem_bnd=10,
)

# POST PROCESS SETTING
DETECTION_POST = dict(iou_threshold=0, sigma=0.35)  # soft nms
