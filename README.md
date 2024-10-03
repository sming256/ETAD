# ETAD: A Unified Framework for Efficient Temporal Action Detection
This repo holds the official implementation of paper: 
["ETAD: A Unified Framework for Efficient Temporal Action Detection"](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Liu_ETAD_Training_Action_Detection_End_to_End_on_a_Laptop_CVPRW_2023_paper.pdf), which is accepted in CVPR workshop 2023.

> Temporal action detection (TAD) with end-to-end training often suffers from the pain of huge demand for computing resources due to long video duration. In this work, we propose an efficient temporal action detector (ETAD) that can train directly from video frames with extremely low GPU memory consumption. Our main idea is to minimize and balance the heavy computation among features and gradients in each training iteration. We propose to sequentially forward the snippet frame through the video encoder, and backward only a small necessary portion of gradients to update the encoder. To further alleviate the computational redundancy in training, we propose to dynamically sample only a small subset of proposals during training. Moreover, various sampling strategies and ratios are studied for both the encoder and detector. ETAD achieves state-of-the-art performance on TAD benchmarks with remarkable efficiency. On ActivityNet-1.3, training ETAD in 18 hours can reach 38.25% average mAP with only 1.3 GB memory consumption per video under end-to-end training.

## Updates
- 12/03/2023: We have released our code and pretrained models for the ActivityNet experiments.

## Installation

**Step 1.** Clone the repository
```
git clone git@github.com:sming256/ETAD.git
cd ETAD
```

**Step 2.** Install PyTorch=2.0.1, Python=3.10.12, CUDA=11.8

```
conda create -n etad python=3.10.12
source activate etad
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Step 3.** Install mmaction2 for end-to-end training
```
pip install openmim
mim install mmcv==2.0.1
mim install mmaction2==1.1.0
pip install numpy==1.23.5
```

## To Reproduce Our Results on ActivityNet 1.3

### End-to-End Experiment

**Download the ActivityNet videos**
- Note that we are not allowed to redistribute the videos without license agreement. You can download the activitynet raw videos from [official website](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform).
- We downsample the videos to 15 fps and resize the shorter side to 256. If you find it's hard to prepare the videos, you can send an email to shuming.liu@kaust.edu.sa to get the videos under license agreements.
- Change the [VIDEO_PATH](configs/anet/e2e_anet_tsp_snippet0.3.py#L26) to the path of your videos.

**Download the backbone weights**
- Download the pretrained [weights](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth) for R(2+1)D backbone and move it to `pretrained/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth`.

**Training**
- `python tools/train.py configs/anet/e2e_anet_tsp_snippet0.3.py 1`
- 1 means using 1 gpu to train.
- The end-to-end experiment takes 18 hours and no more than 10 GB memory for training.
  
**Inference**
- `python tools/test.py configs/anet/e2e_anet_tsp_snippet0.3.py 1`
- The testing takes around 45 mins.

**Evaluation**
- `python tools/post.py configs/anet/e2e_anet_tsp_snippet0.3.py`

### Feature-based Experiment

**Download the TSP features**
- You can download TSP feature from [ActionFormer](https://github.com/happyharrycn/actionformer_release#to-reproduce-our-results-on-activitynet-13), or directly from this [Google drive](https://drive.google.com/file/d/1VW8px1Nz9A17i0wMVUfxh6YsPCLVqL-S/view?usp=sharing).
- Change the [FEATURE_PATH]([configs/anet/feature_anet_tsp.py#L7) to the path of your features.

**Training**
- `python tools/train.py configs/anet/feature_anet_tsp.py 1`
- The feature-based experiment is fast (6 mins in my workstation).

**Testing and Evaluation**
- `python tools/test.py configs/anet/feature_anet_tsp.py 1 && python tools/post.py configs/anet/feature_anet_tsp.py`


### Pretrained Models
You can download the pretrained models in this [link](https://github.com/sming256/ETAD/releases/).
If you want to do inference with our checkpoint, you can simply run

```
python tools/test.py configs/anet/e2e_anet_tsp_snippet0.3.py 1 --checkpoint e2e_anet_snippet0.3_bs4_92e98.pth.tar
python tools/post.py configs/anet/e2e_anet_tsp_snippet0.3.py
```

The results on ActivityNet (with CUHK classifier) should be

| mAP at tIoUs         | 0.5   | 0.75  | 0.95  | Avg   |
| -------------------- | ----- | ----- | ----- | ----- |
| ETAD - TSP - Feature | 54.96 | 39.06 | 9.21  | 37.80 |
| ETAD - TSP - E2E     | 56.22 | 39.93 | 10.23 | 38.73 |

You can also download our **logs, and results** from [Google Drive](https://drive.google.com/drive/folders/1K4woHQn1FvODSp9UPsoC5Ac0NMhIzKS-?usp=sharing). 


## Contact
If you have any questions about our work, please contact Shuming Liu (shuming.liu@kaust.edu.sa).

## References
If you are using our code, please consider citing our paper.
```
@inproceedings{liu2023etad,
  title={ETAD: Training Action Detection End to End on a Laptop},
  author={Liu, Shuming and Xu, Mengmeng and Zhao, Chen and Zhao, Xu and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4524--4533},
  year={2023}
}
```

If you are using TSP features, please cite
```
@inproceedings{alwassel2021tsp,
  title={{TSP}: Temporally-sensitive pretraining of video encoders for localization tasks},
  author={Alwassel, Humam and Giancola, Silvio and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={3173--3183},
  year={2021}
}
```
