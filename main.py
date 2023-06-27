from handDetector import handDetector

import cv2
import time
import joblib
import pandas as pd
import pyautogui

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    model = joblib.load('model/model.pkl')
    currentGesture = None

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Mirror horizontally
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        screen_width, screen_height = pyautogui.size()
        
        if len(lmlist) != 0:
            for i in range(0, len(lmlist)):
                if i != 0: lmlist[i] = lmlist[i][0], lmlist[i][1]-lmlist[0][1], lmlist[i][2]-lmlist[0][2]

            data = [lmlist[1][1], lmlist[1][2], lmlist[2][1], lmlist[2][2], lmlist[3][1], lmlist[3][2], lmlist[4][1], lmlist[4][2], lmlist[5][1], lmlist[5][2], lmlist[6][1], lmlist[6][2], lmlist[7][1], lmlist[7][2], lmlist[8][1], lmlist[8][2], lmlist[9][1], lmlist[9][2], lmlist[10][1], lmlist[10][2], lmlist[11][1], lmlist[11][2], lmlist[12][1], lmlist[12][2], lmlist[13][1], lmlist[13][2], lmlist[14][1], lmlist[14][2], lmlist[15][1], lmlist[15][2], lmlist[16][1], lmlist[16][2], lmlist[17][1], lmlist[17][2], lmlist[18][1], lmlist[18][2], lmlist[19][1], lmlist[19][2], lmlist[20][1], lmlist[20][2]]

            gesture = model.predict([data])[0]
            if gesture != currentGesture:
                print(gesture)

            if gesture == 'pointing':
                index_finger_x, index_finger_y = lmlist[8][1], lmlist[8][2]
                cursor_x = int(index_finger_x * screen_width / img.shape[1])
                cursor_y = int(index_finger_y * screen_height / img.shape[0])
            
                # Move the cursor
                pyautogui.moveTo(cursor_x, cursor_y)

            if gesture == 'fingerGun':
                pyautogui.click()

            currentGesture = gesture if gesture != 'none' else currentGesture

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()