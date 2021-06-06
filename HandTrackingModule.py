import cv2
import mediapipe as mp
import time
import math

"""
Main class housing de detector and landmark processing functions. Import tis for use in another file.
"""
class handDetector():

    """
    Setting up the tracking parameters as well as starting the tracking. and drawing.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    """
    imports and transforms the image into the rigth colour space and then tries to find the hand landmarks.
    """
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    """
    extracts the position fo the handmarks and translates them to a pixel coordanet space. It also computes a boundign box.
    """
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) == 1:
                myHand = self.results.multi_hand_landmarks[handNo]

                h, w, c = img.shape

                for id, lm in enumerate(myHand.landmark):

                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    xList.append(cx)
                    yList.append(cy)
                    if draw:
                        cv2.circle(img, (cx, cy), 5, [255, 0, 255], cv2.FILLED)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                if draw:
                    cv2.rectangle(img, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (0, 255, 0), 5)

        return self.lmList, bbox



    """
    Returns the distance between the points p1 and p2 wich are landmarks.
    """
    def findDistance(self, img, p1, p2, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

"""
A example implementation on the tracking.
"""
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = handDetector()

    while True:
        sucsess, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
