from handDetector import handDetector
import cv2
import time
import os
import csv

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    gestures = ['pointing', 'fingerGun']

    csv_file = './logs/data.csv'
    csv_header = ['gesture', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y', '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x', '11y', '12x', '12y', '13x', '13y', '14x', '14y', '15x', '15y', '16x', '16y', '17x', '17y', '18x', '18y', '19x', '19y', '20x', '20y']

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    with open(csv_file, 'a', newline='') as file:
        if file.tell() == 0:
            writer = csv.writer(file)
            writer.writerow(csv_header)

    for gesture in gestures:
        # Show gesture on the webcam window
        success, img_gesture = cap.read()
        cv2.putText(img_gesture, gesture, (150, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Gesture", img_gesture)
        cv2.waitKey(5000)
        cv2.destroyWindow("Gesture")

        for _ in range(1000):
            success, img = cap.read()
            img = cv2.flip(img, 1)  # Mirror horizontally
            img = detector.findHands(img)
            lmlist = detector.findPosition(img)

            if len(lmlist) != 0:
                for i in range(0, len(lmlist)):
                    if i != 0:
                        lmlist[i] = lmlist[i][0], lmlist[i][1] - lmlist[0][1], lmlist[i][2] - lmlist[0][2]

                data = [lmlist[1][1], lmlist[1][2], lmlist[2][1], lmlist[2][2], lmlist[3][1], lmlist[3][2],
                        lmlist[4][1], lmlist[4][2], lmlist[5][1], lmlist[5][2], lmlist[6][1], lmlist[6][2],
                        lmlist[7][1], lmlist[7][2], lmlist[8][1], lmlist[8][2], lmlist[9][1], lmlist[9][2],
                        lmlist[10][1], lmlist[10][2], lmlist[11][1], lmlist[11][2], lmlist[12][1], lmlist[12][2],
                        lmlist[13][1], lmlist[13][2], lmlist[14][1], lmlist[14][2], lmlist[15][1], lmlist[15][2],
                        lmlist[16][1], lmlist[16][2], lmlist[17][1], lmlist[17][2], lmlist[18][1], lmlist[18][2],
                        lmlist[19][1], lmlist[19][2], lmlist[20][1], lmlist[20][2]]

                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([gesture] + data)

                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

                cv2.imshow("Image", img)
                cv2.waitKey(1)
            else:
                continue

if __name__ == "__main__":
    main()
