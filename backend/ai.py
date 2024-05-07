import cv2

BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

BODY_PARTS_LIST = list(BODY_PARTS.keys())

class SampleOpenPose:

    def __init__(self) -> None:
        proto = "./models/pose_deploy_linevec_faster_4_stages.prototxt"
        weights = "./models/pose_iter_160000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)

    async def pretreatment(self, frame):
        size = 3
        frame = cv2.resize(frame, (64*size,48*size))
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        return self._grid_poseLine(output=output,frame=frame)

    def _grid_poseLine(self,output, frame):
        skeletron = frame
        height, width, _ = frame.shape

        H = output.shape[2]
        W = output.shape[3]

        points = []
        skeletron_point = []
        for i in range(0,15):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            x = (width * point[0]) / W
            y = (height * point[1]) / H

            if prob > 0.1:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
                skeletron_point.append({
                    "body_parts":BODY_PARTS_LIST[i],
                    "prob":prob,
                    "point":{int(x),int(y)}
                })
            else:
                points.append(None)

        for pair in POSE_PAIRS:
            first = pair[0]
            first = BODY_PARTS[first]
            second = pair[1]
            second = BODY_PARTS[second]
            if points[first] and points[second]:
                cv2.line(skeletron, points[first], points[second], (0, 255, 0), 2)

        skeletron_str = '[' + ', '.join(str(d) for d in skeletron_point) + ']'
        return skeletron, skeletron_str