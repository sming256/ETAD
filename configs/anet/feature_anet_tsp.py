EXP_NAME = "feature_anet_tsp"

E2E_SETTING = dict(mode=False)

# DATASET SETTING
DATASET = dict(name="anet_1_3", tscale=128, dscale=128)
FEATURE = dict(path="data/anet/features/tsp_features", online_resize=True)

# MODEL SETTINGS
MODEL = dict(in_channels=512, roi_size=24, stage=[0.7, 0.8, 0.9], extend_ratio=0.5)

# SAMPLING SETTINGS
SAMPLING_RATIO = dict(snippet=0, proposal=0.06)  # set snippet=0 for all feature based experiment
SAMPLING_STRATEGY = dict(proposal="random")

# SOLVER SETTING
SOLVER = dict(
    tal_lr=1.0e-3,
    backbone_lr=0,
    step_size=5,
    gamma=0.1,
    batch_size=16,
    workers=8,
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
