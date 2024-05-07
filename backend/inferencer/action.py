import torch
import torch.nn as nn

import numpy as np

import mmengine

from typing import List,Any
'''
from mmaction.apis import init_recognizer,inference_recognizer

class Action:
    def __init__(self,
                 config:str,
                 checkpoint:str
                 ) -> None:
        self.config = config
        self.checkpoint = checkpoint
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
    
    def init_model(self)->nn.Module:
        config = mmengine.Config.fromfile(self.config)
        model = init_recognizer(config,self.checkpoint,self.device)
        return model
    
    def recognize(self,model:nn.Module,frame:np.ndarray,pose_results:List[Any],num_frame:int):
        h,w = frame.shape[:2]
        
        fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
        
        num_person = max([len(x['keypoints']) for x in pose_results])

        num_keypoint = 17
        keypoint = np.zeros((num_frame, num_person, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_frame, num_person, num_keypoint),
                                dtype=np.float16)
        for i, poses in enumerate(pose_results):
            keypoint[i] = poses['keypoints']
            keypoint_score[i] = poses['keypoint_scores']

        fake_anno['keypoint'] = keypoint.transpose((1, 0, 2, 3))
        fake_anno['keypoint_score'] = keypoint_score.transpose((1, 0, 2))
        
        
        results = inference_recognizer(model,fake_anno)
        return results
        
'''