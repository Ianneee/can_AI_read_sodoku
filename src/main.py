from errorsClass import AnglesNotFound
import getSquares as gs
import cv2
import sys
import tkinter as tk
from tkinter import filedialog
from numbersRecognizer import NumbersRecognizer

camera_win = 'WebCam'
SODOKU_SIZE = 540

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    print("Selected file:", file_path)


def display_boxes(boxs, i):
    cv2.imshow('Results', boxs[i])
    cv2.displayStatusBar("Results", 'Navigation n: next | p: previous', 0)


def draw_numbers(predictions, sodoku_img):
    s = SODOKU_SIZE / 9
    for i in range(9):
        for j in range(9):
            q = predictions[i * 9 + j]
            if q:
                x = int(j * s + 5)
                y = int(i * s + 30)
                cv2.putText(sodoku_img,str(q),(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),1)
                cv2.waitKey(1)

    cv2.imshow("Sudoku!", sodoku_img)
    cv2.waitKey(1)
    cv2.displayStatusBar("Sudoku!", 'Press k for close this window', 0)


def main():
    if len(sys.argv) == 1:
        cam = 0
    else:
        cam = int(sys.argv[1])

    rec = NumbersRecognizer()

    root = tk.Tk()
    root.withdraw()

    video = cv2.VideoCapture(cam)

    boxs = None
    i = 0

    while True:
        _, frame = video.read()

        cv2.imshow(camera_win, frame)
        cv2.displayStatusBar(camera_win, 'f: open file - s: take snapshot - q: quit', 0)

        k = cv2.waitKey(30)
        try:
            if k == ord('q'):
                exit(0)

            elif k == ord('s'):
                boxs, warped = gs.getSquares(video=video)
                i = 0
                rec.recognize(boxs)
                boxs = rec.cleaned_data
                display_boxes(boxs, i)
                draw_numbers(rec.predictions, warped)

            elif k == ord('f'):
                file_path = filedialog.askopenfilename()
                boxs, warped = gs.getSquares(img_path=file_path)
                i = 0
                rec.recognize(boxs)
                boxs = rec.cleaned_data
                display_boxes(boxs, i)
                draw_numbers(rec.predictions, warped)

            elif k == ord('n'):
                i = (i + 1) % 81
                display_boxes(boxs, i)

            elif k == ord('p'):
                i = (i - 1) % 81
                display_boxes(boxs, i)

            elif k == ord('k'):
                cv2.destroyWindow("Sudoku!")

        except AnglesNotFound:
            cv2.displayOverlay(camera_win, "Can't detected the sudoku! Please try again", 2000)


if __name__ == "__main__":
    main()
