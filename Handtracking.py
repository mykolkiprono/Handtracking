import mediapipe as mp 
import cv2 as cv
import time


class HandDetector():
    def __init__(self, mode = False, max_hands = 2, detection_confidence = 0.7, track_confdence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confdence = track_confdence

        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(self.mode, max_hands, self.detection_confidence, self.track_confdence)
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw = True):
        imagergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        self.results = self.hands.process(imagergb)

        if self.results.multi_hand_landmarks:

            for handlms in self.results.multi_hand_landmarks:
                if draw:                                
                    self.mpdraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findpos(self, frame, handno, draw = False):
        lmlist  = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for lm_id, lm in enumerate(myhand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                lmlist.append([lm_id, cx, cy])
                if draw:
                     cv.circle(frame, (cx, cy), 5, (255,0,0), cv.FILLED)
        return lmlist

   
def main():
    ptime = 0
    ctime = 0
    detector = HandDetector()

    cap = cv.VideoCapture(0)


    while True:
        is_True, frame = cap.read()

        imagergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        detector.findHands(frame, draw = False)
        lmlist = detector.findpos(frame, 0, draw = False)

        if len(lmlist) != 0:
            print(lmlist[4])

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 3)
        cv.imshow('frame',frame)

        cv.waitKey(1)



if __name__ == '__main__':
    main()
