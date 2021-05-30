import cv2
import time
import numpy as np
import HandTrackingModule as htm

wCam, hCam = 1920, 1080


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0


detector = htm.handDetector()

while True:
    sucsess, img = cap.read()

    img = detector.findHands(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)