import cv2 as cv
import mediapipe as mp 
import time
import Handtracking as ht

ptime = 0
ctime = 0
detector = ht.HandDetector()
cap = cv.VideoCapture(0)

while True:
        is_True, frame = cap.read()

        imagergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        detector.findHands(frame)
        lmlist = detector.findpos(frame, 0)

        if len(lmlist) != 0:
            print(lmlist[4])

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 3)
        cv.imshow('frame',frame)

        cv.waitKey(1)