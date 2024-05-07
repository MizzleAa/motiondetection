import numpy as np

import torch
import torch.nn as nn

from typing import List,Dict,Any

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.structures import DetDataSample
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                        '`init_detector` from `mmdet.apis`. These apis are '
                        'required in this inference api! ')

class Detection:
    def __init__(self,
                 config:str,
                 checkpoint:str
                 ) -> None:
        """Detection init

        Args:
            config (str): 디텍션 모델 config 파일
            checkpoint (str): 디텍션 모델 weight 파일
        """
        self.config = config
        self.checkpoint = checkpoint
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        
    def init_model(self) ->nn.Module:
        """디텍션 모델 생성

        Returns:
            nn.Module: 디텍션 모델
        """
        model = init_detector(config=self.config, checkpoint=self.checkpoint, device=self.device)
        return model
        
    def detect(self,model:nn.Module,frame:np.ndarray,det_cat_id:int=0,det_score_thr:float=0.9)->np.ndarray:
        """detection 수행

        Args:
            model (nn.Module): 디텍션 모델
            frame (np.ndarray): 디텍션할 타겟 이미지
            det_cat_id (int, optional): 디텍션 카테고리 0인 경우 사람. Defaults to 0.
            det_score_thr (float, optional): 디텍션 스코어 임계값. Defaults to 0.9.

        Returns:
            np.ndarray : 디텍션 결과
        """
        det_data_sample: DetDataSample = inference_detector(model, frame)
        pred_instance = det_data_sample.pred_instances.cpu().numpy()
        bboxes = pred_instance.bboxes
        scores = pred_instance.scores
        
        valid_idx = np.logical_and(pred_instance.labels == det_cat_id,
                                   pred_instance.scores > det_score_thr)
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]

        bboxes = np.concatenate((bboxes, scores[:, None]), axis=-1)
        return bboxes
    
    def extract_results(self,results:np.ndarray)->List[Dict[str,Any]]:
        """디텍션 결과 정리

        Args:
            results (np.ndarray): 디텍션 결과 배열

        Returns:
            List[Dict[str,Any]]: 디텍션결과를 dictionary 형태로 정리한 결과
        """
        det_results = []
        
        for result in results:
            bbox = list(map(int,result[:4]))
            score = result[-1]
            box = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
            det_results.append(
                dict(
                    bbox = box,
                    score = score
                )
            )
        return det_results