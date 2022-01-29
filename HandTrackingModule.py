import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode  # create an object and that object will have a variable
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils  # draws the dots/joints
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # draw for a single hand
        return img

    def findPosition(self, img, handNo=0, draw = True):
        # get id number (tracking dot index) and landmark information to differentiate each hand:
        lmList = [] #store a landmark list
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                # convert the decimals to pixel values:
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # position off center
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                # # draw a circle on the first indices (id numbered 1)
                # if id == 0:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                # if id == 4:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)  # camera
    detector = handDetector()
    while True:  # processing each frame
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        #calculate frames:
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # calculate frames per second
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255),
                    3)  # on top of object, number converted to integer, position, font, size, color, thickness
        cv2.imshow("Image", img)
        cv2.waitKey(1)  # camera detects at least every 1ms


if __name__ == "__main__":
    main()
