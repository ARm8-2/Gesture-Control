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

    pyautogui.FAILSAFE = False

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Mirror horizontally
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        rellmlist = detector.findPosition(img)
        screen_width, screen_height = pyautogui.size()

        if len(lmlist) != 0:
            for i in range(0, len(lmlist)):
                if i != 0:
                    rellmlist[i] = lmlist[i][0], lmlist[i][1]-lmlist[0][1], lmlist[i][2]-lmlist[0][2]

            data = [rellmlist[1][1], rellmlist[1][2], rellmlist[2][1], rellmlist[2][2], rellmlist[3][1], rellmlist[3][2], rellmlist[4][1], rellmlist[4][2], rellmlist[5][1], rellmlist[5][2], rellmlist[6][1], rellmlist[6][2], rellmlist[7][1], rellmlist[7][2], rellmlist[8][1], rellmlist[8][2], rellmlist[9][1], rellmlist[9][2], rellmlist[10][1], rellmlist[10][2], rellmlist[11][1], rellmlist[11][2], rellmlist[12][1], rellmlist[12][2], rellmlist[13][1], rellmlist[13][2], rellmlist[14][1], rellmlist[14][2], rellmlist[15][1], rellmlist[15][2], rellmlist[16][1], rellmlist[16][2], rellmlist[17][1], rellmlist[17][2], rellmlist[18][1], rellmlist[18][2], rellmlist[19][1], rellmlist[19][2], rellmlist[20][1], rellmlist[20][2]]

            gesture = model.predict([data])[0]
            currentGesture = gesture if gesture != 'none' else currentGesture

            if currentGesture == 'pointing' or currentGesture == 'fingerGun':
                index_finger_x, index_finger_y = lmlist[8][1], lmlist[8][2]
                cursor_x = int(index_finger_x * screen_width / img.shape[1])
                cursor_y = int(index_finger_y * screen_height / img.shape[0])

                if gesture == 'fingerGun':
                    pyautogui.mouseDown()
                else:
                    pyautogui.mouseUp()
                
                pyautogui.moveTo(cursor_x*1.1, (cursor_y-100)*1.1)

            if currentGesture == 'thumbOut':
                pyautogui.click()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
