import torch
import torch.nn as nn

import numpy as np

from typing import List,Dict,Any,Tuple

try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import PoseDataSample, merge_data_samples
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_topdown` and '
                        '`init_model` from `mmpose.apis`. These apis '
                        'are required in this inference api! ')


class Pose:
    def __init__(self,
                 config:str,
                 checkpoint:str
                 ) -> None:
        """Pose init

        Args:
            config (str): 모델 config 파일
            checkpoint (str): 모델 weight 파일
        """
        self.config = config
        self.checkpoint = checkpoint
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        
    def init_model(self) ->nn.Module:
        """pose 모델 생성

        Returns:
            nn.Module: pose 모델
        """
        model = init_model(config=self.config, checkpoint=self.checkpoint, device=self.device)
        return model
        
    def estimate(self,model:nn.Module,frame:np.ndarray,det_result:List[np.ndarray])->Tuple[Dict[str,Any],List[List[np.ndarray]]]:
        """_summary_

        Args:
            model (nn.Module): 포즈 모델
            frame (np.ndarray): 포즈 추정할 이미지
            det_result (List[np.ndarray]): detection 결과

        Returns:
            Tuple[Dict[str,Any],List[List[np.ndarray]]]: 추정결과
        """
        pose_data_samples: List[PoseDataSample] = inference_topdown(model, frame, det_result[..., :4], bbox_format='xyxy')
        pose_data_sample = merge_data_samples(pose_data_samples)
        pose_data_sample.dataset_meta = model.dataset_meta
        poses = pose_data_sample.pred_instances.to_dict()

        keypoints = poses["keypoints"]
        keypoint_scores = poses["keypoint_scores"]

        pose_results=[]
        for keypoint,keypoint_score in zip(keypoints,keypoint_scores):
            pose_result = np.concatenate((keypoint, keypoint_score[:,None]), axis=-1)
            pose_results.append(pose_result)
        return poses,pose_results