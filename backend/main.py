from typing import Any, Dict, List

from ai import SampleOpenPose
from fastapi import FastAPI, WebSocket

import datetime
import cv2
import base64
import numpy as np
import json
import torch
import copy

from inferencer.detection import Detection
from inferencer.pose import Pose
from inferencer.frame_action import FrameAction
from core.config import *

app = FastAPI()

detection = Detection(DET_CONFIG,DET_CHECKPOINT)
detection_model = detection.init_model()
#pose model init
# pose = Pose(POSE_CONFIG,POSE_CHECKPOINT)
# pose_model = pose.init_model()

action = FrameAction(ACTION_CONFIG,ACTION_CHECKPOINT,ACTION_LABEL)
action_model = action.init_model()
action_pipeline = action.get_pipeline(action_model)

@app.websocket("/video/{id}")
async def video(websocket: WebSocket, id:int):
    try:
        await websocket.accept()
        cap = cv2.VideoCapture(id)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_cnt = 0
        action_label="waiting action"
        step = 15
        frames = []
        while True:
            try:
                start_time = datetime.datetime.now().timestamp() 
                information = ""
                ret, frame = cap.read()
                # visual = copy.copy(frame)
                frame_cnt+=1
                frames.append(frame)

                if not ret:
                    break

                _, buffer = cv2.imencode('.jpg', frame)
                image = base64.b64encode(buffer).decode("utf-8")
                ########################
                #ai
                det_results = detection.detect(detection_model,frame)
                ex_det_results = detection.extract_results(det_results)

                torch.cuda.empty_cache()
                
                if len(frames)%step==0:
                    action_result = action.recognize(action_model,frames,action_pipeline)
                    action_label = action.extract_result(action_result)
                    torch.cuda.empty_cache()
                    frames=[]
                # _,pose_results = pose.estimate(pose_model,frame,det_results)
                # visual, information = predict(visual, ex_det_results, pose_results,0.5)
                visual, error = predict2(frame, ex_det_results ,action_label)
                
                _, buffer = cv2.imencode('.jpg', visual)
                visual = base64.b64encode(buffer).decode("utf-8")
                ########################
                end_time = datetime.datetime.now().timestamp() 
                fps = str(1.0/(end_time-start_time))
                data = {"origin_image": image, "skeletron_image": visual, "information":str(action_label), "fps":fps, "error":error}
                json_data = json.dumps(data)
                await websocket.send_text(json_data)

            except Exception as e:
                print(e)
                break

        cap.release()
    except:
        pass


def predict(frame, ex_det_results,pose_results, key_threshold):

    information = []
    for det_res,pose_res in zip(ex_det_results,pose_results):
        bbox = det_res["bbox"]
        score = round(det_res["score"],3)
        # score 넣기
        text_score = str(score)
        text_size, _ = cv2.getTextSize(text_score, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, (bbox[0], bbox[1]-text_h, text_w, text_h), (0, 0, 0), -1)
        cv2.putText(frame, text_score, (bbox[0], bbox[1] + 10-text_h), cv2.FONT_HERSHEY_DUPLEX, 0.5, (72, 101, 241),
                                1, cv2.LINE_AA)
        # detection 결과 박스
        cv2.rectangle(frame, bbox, color=(255,0,0), thickness=1)

        #pose keypoint 좌표 찍기
        pose_list = []
        for res in pose_res:
            if res[-1]>key_threshold:
                cv2.circle(frame,list(map(int,list(res[:2]))),4,(0,255,255),thickness=-1,lineType=cv2.FILLED)
                pose_list.append(res[:2].tolist())
        #pose keypoint 좌표 연결하기

        for line in LINES:
            keypoint_1 = pose_res[line[0]]
            keypoint_2 = pose_res[line[1]]
            if keypoint_1[-1]>key_threshold and keypoint_2[-1]>key_threshold:
                cv2.line(frame, tuple(map(int,keypoint_1[:2])), tuple(map(int,keypoint_2[:2])), (255, 255, 0), 1)

        information.append(
            {
                "bbox":bbox,
                "score":score,
                "pose_list":pose_list
            }
        )
    return frame, information


def predict2(frame:np.ndarray,
                det_results:List[Dict[str,Any]],
                action_label:str=None)->np.ndarray:
    """결과 시각화

    Args:
        frame (np.ndarray): 이미지
        det_results (List[Dict[str,Any]]): 디텍션 결과
        action_label (str): 액션 판별 결과 레이블
    Returns:
        np.ndarray: 시각화된 이미지
        pose_results(n,17,3)
    """
    error = False
    for det_res in det_results:
        bbox = det_res["bbox"]
        score = det_res["score"]
        score = round(score,3)
        # score 넣기
        # text_label = f"{action_label}|{score:.2f}"
        text_label = f"Person : {score:.2f}"

        text_size, _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, (bbox[0], bbox[1]-text_h, text_w, text_h), (0, 0, 0), -1)
        cv2.putText(frame, text_label, (bbox[0], bbox[1] + 10-text_h), cv2.FONT_HERSHEY_DUPLEX, 0.5, (72, 101, 241), 1, cv2.LINE_AA)
        # detection 결과 박스
        cv2.rectangle(frame, bbox, color=(255,0,0), thickness=1)

    ven_list = ["squat","running on treadmill","push up", "stretching leg", "wrapping present"]
    if action_label in ven_list:
        error = True
    return frame, error