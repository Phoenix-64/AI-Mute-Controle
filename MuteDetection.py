import cv2
import time
import math
import numpy as np
import HandTrackingModule as htm
import pandas as pd
import statistics
import voicemeeter

kind = 'potato'
voicemeeter.launch(kind, delay=0.125)
with voicemeeter.remote(kind, delay=0.125) as vmr:
    pass

wCam, hCam = 640, 480

cap = cv2.VideoCapture(2)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

detector = htm.handDetector()

averages = []

send = False
fingers = [8, 12, 16, 20]
area = 0
while True:
    sucsess, img = cap.read()

    img = detector.findHands(img, draw=False)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        print(send)

        if not send:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

            if 100 < area < 2000:

                for i in fingers:
                    lengths = []
                    length, img, pointData = detector.findDistance(img, 0, i, draw=False)
                    # print(length)
                    lengths.append(length)

                averages.append(statistics.mean(lengths))
                # print(statistics.mean(lengths))
                if len(averages) > 10:
                    #print(statistics.mean(averages))

                    if 50 < statistics.mean(averages) < 200:
                        vmr.inputs[0].mute = True
                        send = True

                    elif 200 < statistics.mean(averages):
                        vmr.inputs[0].mute = False
                        send = True

                    del averages[0:2]
    else:
        send = False
        averages = []



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    #cv2.imshow("Img", img)
    cv2.waitKey(1)
