from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = f"{BASE_DIR}/models/checkpoint"

DET_CONFIG = f"{CHECKPOINT_ROOT}/detection/faster-rcnn_r50_fpn_2x_coco_infer.py"
DET_CHECKPOINT = f"{CHECKPOINT_ROOT}/detection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"

# POSE_CONFIG = f"{CHECKPOINT_ROOT}/pose/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py"
# POSE_CHECKPOINT = f"{CHECKPOINT_ROOT}/pose/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

POSE_CONFIG = f"{CHECKPOINT_ROOT}/pose/rtmpose-t_8xb256-420e_coco-256x192.py"
POSE_CHECKPOINT = f"{CHECKPOINT_ROOT}/pose/rtmpose-t_8xb256-420e_coco-256x192.pth"

ACTION_CONFIG = f"{CHECKPOINT_ROOT}/action/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py"
ACTION_CHECKPOINT = f"{CHECKPOINT_ROOT}/action/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
ACTION_LABEL = f"{CHECKPOINT_ROOT}/action/label_map_k400.txt"

LINES=[
    [0,1],[1,2],[2,0],[1,3],[3,5],[2,4],[4,6],
    [5,6],[5,7],[7,9],[5,11],[11,12],[6,8],[8,10],[6,12],
    [12,14],[14,16],[11,13],[13,15]
]