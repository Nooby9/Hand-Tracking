import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)  #camera

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  #draws the dots/joints

#fps:
pTime = 0
cTime = 0

while True: #processing each frame
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #get id number (tracking dot index) and landmark information to differentiate each hand:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                # convert the decimals to pixel values:
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) #position off center
                print(id, cx, cy)
                #draw a circle on the first indices (id numbered 1)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #draw for a single hand

    cTime = time.time()
    fps = 1/(cTime-pTime) #calculate frames per second
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3) #on top of object, number converted to integer, position, font, size, color, thickness
    cv2.imshow("Image", img)
    cv2.waitKey(1) #camera detects at least every 1ms
