import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

#using the module demonstration
pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)  # camera
detector = htm.handDetector()
while True:  # processing each frame
    success, img = cap.read()
    img = detector.findHands(img) #lines
    lmList = detector.findPosition(img, draw=False) #dots
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