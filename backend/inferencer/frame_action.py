import torch
import torch.nn as nn

import numpy as np

from typing import List,Any

from mmengine.dataset import Compose,pseudo_collate
from mmengine.registry import init_default_scope
from mmaction.structures import ActionDataSample
from mmaction.apis import init_recognizer

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

class FrameAction:
    def __init__(self,
                 config:str,
                 checkpoint:str,
                 label:str,
                 ) -> None:
        self.config = config
        self.checkpoint = checkpoint
        self.label = label
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        self.label_map = [x.strip() for x in open(self.label).readlines()]
        self.num_clips = None
        self.clip_len = None
        
    def init_model(self)->nn.Module:
        model = init_recognizer(self.config,self.checkpoint,self.device)
        return model

    def get_pipeline(self,model:nn.Module)->Compose:
        """파이프라인 생성

        Args:
            model (nn.Module): 액션 모델

        Returns:
            Compose: 파이프라인
        """
        init_default_scope('mmaction')
        # prepare test pipeline from non-camera pipeline
        cfg = model.cfg
        pipeline = cfg.test_pipeline
        pipeline_ = pipeline.copy()
        for step in pipeline:
            if 'SampleFrames' in step['type']:
                self.num_clips = step['num_clips']
                self.clip_len = step['clip_len']
                pipeline_.remove(step)
            if step['type'] in EXCLUED_STEPS:
                # remove step to decode frames
                pipeline_.remove(step)
        test_pipeline = Compose(pipeline_)
        return test_pipeline
    
    def recognize(self,
                model:nn.Module,
                frames:List[np.ndarray],
                pipeline:Compose)->ActionDataSample:
    
        data = dict(img_shape=None, modality='RGB', label=-1)
        if self.num_clips:
            data['num_clips'] = self.num_clips
        if self.clip_len:
            data['clip_len'] = self.clip_len
        data["imgs"] = frames
        data["img_shape"] = frames[0].shape[:2]
        
        p_data = pipeline(data)
        cur_data = pseudo_collate([p_data])
        
        with torch.no_grad():
            result = model.test_step(cur_data)[0]
        
        return result
    
    def extract_result(self,result:np.ndarray)->str:
        """액션 결과 추출

        Args:
            result (np.ndarray): 액션 결과 배열

        Returns:
            str: 액션 레이블
        """
        max_pred_index = result.pred_scores.item.argmax().item()
        action_label = self.label_map[max_pred_index]
        return action_label