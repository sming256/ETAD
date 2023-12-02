# anet
from .anet_1_3.dataset import VideoDataSet as anet_1_3_dataset
from .anet_1_3.post_det import detection_post as anet_1_3_post_det

# thumos
# from .thumos_14.dataset import VideoDataSet as thumos_14_dataset
# from .thumos_14.post_det import detection_post as thumos_14_post_det


# hacs
# from .hacs.dataset import VideoDataSet as hacs_dataset
# from .hacs.post_det import detection_post as hacs_post_det


def build_dataset(name, **kwargs):
    if name == "anet_1_3":
        return anet_1_3_dataset(**kwargs)


def build_post_processing(name, **kwargs):
    if name == "anet_1_3":
        return anet_1_3_post_det(**kwargs)
