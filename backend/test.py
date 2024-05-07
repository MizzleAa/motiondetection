import cv2

from ai import SampleOpenPose

sampleAi = SampleOpenPose()

frame = cv2.imread("./images/1.jpg")
# frame = cv2.resize(frame,(368,368))
skeletron_frame, point = sampleAi.pretreatment(frame)

cv2.imshow("skeletron_frame",skeletron_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


print(point)