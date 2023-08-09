from errorsClass import AnglesNotFound
import getSquares as gs
import cv2
import sys

def main():
    if len(sys.argv) == 1:
        cam = 0
    else:
        cam = int(sys.argv[1])

    video = cv2.VideoCapture(cam)
    #video.set(cv2.CAP_PROP_AUTOFOCUS, 12)
    while True:
        _, frame = video.read()

        cv2.imshow('frame', frame)

        k = cv2.waitKey(30)
        if k == ord('q'):
            exit(0)
        elif k == ord('s'):
            try:
                #q = gs.getSquares(img_path="last.jpg")
                #q = gs.getSquares(img_path="Sudoku.jpg")
                gs.getSquares(video=video)
            #if len(q) != 0:
            #    cv2.imshow("grid",q[0])
            #    cv2.waitKey(0)
            #else:
            #    print("Niente")
            except AnglesNotFound:
                print("Ripeti la foto")



if __name__ == "__main__":
    main()
