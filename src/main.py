from errorsClass import AnglesNotFound
import getSquares as gs
import cv2
import sys
import tkinter as tk
from tkinter import filedialog

camera_win = 'WebCam'

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    print("Selected file:", file_path)


def main():
    if len(sys.argv) == 1:
        cam = 0
    else:
        cam = int(sys.argv[1])

    root = tk.Tk()
    root.withdraw()

    video = cv2.VideoCapture(cam)
    while True:
        _, frame = video.read()

        cv2.imshow(camera_win, frame)
        cv2.displayStatusBar(camera_win, 'f: open file - s: take snapshot - q: quit', 0)

        k = cv2.waitKey(30)
        try:
            if k == ord('q'):
                exit(0)
            elif k == ord('s'):
                    gs.getSquares(video=video)
            elif k == ord('f'):
                file_path = filedialog.askopenfilename()
                gs.getSquares(img_path=file_path)
            elif k == ord('k'):
                cv2.destroyWindow("Sudoku!")

        except AnglesNotFound:
            cv2.displayOverlay(camera_win, "Can't detected the sudoku! Please try again", 2000)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
