import cv2
import numpy as np
from errorsClass import AnglesNotFound
from typing import Union

#Funzione che ottiene gli 81 quadratini come singole immagini in una lista
def getSquares(video: Union[None, cv2.VideoCapture] = None, img_path: Union[None, str] = None):
        if not video and not img_path:
            raise AttributeError("You need to provide at least one argument.")

        if video:
            _, img = video.read()
        elif img_path:
            img = cv2.imread(img_path)

        img = cv2.resize(img,(1080,940))

        contours = getContours(img) #Contorni

        src_points = getSrcPoints(contours) #Coordinate sudoku
        #src_points = getSrcPoints_threshold(contours) #Coordinate sudoku
        #src_points = getSrcPoints_o(img, contours) #Coordinate sudoku
        if src_points is None:
            raise AnglesNotFound("Can't define the contour of the sodoku")

        warped = getBirdEye(img,src_points) #Sudoku dall'alto
        warped = cv2.resize(warped, (540, 540))

        #ottengo i singoli quadratini del sudoku dividendo in 9 righe uguali l'immagine
        righe = np.vsplit(warped,9)
        quadratini = [] #lista di immagini da restituire
        for row in righe:
                #divido ogni riga creata in 9 quadratini e così iterando otterrò 81 quadratini diversi.
                colonne = np.hsplit(row,9)
                for quadrato in colonne:
                        quadratini.append(quadrato)

        return quadratini, warped


#Funzione che ottiene i contorni dell'immagine passata
def getContours(img):

        #trasformo l'immagine in grayscale (necessario per farla diventare binaria)
        grigio = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #trasformo l'immagine in binaria (nero o bianco)
        # ret, thresh = cv2.threshold(grigio,127,255,0) #la scelta è tra threshold e canny edge detection

        edges = cv2.Canny(grigio,50,100)

        #troviamo i contorni (restituisce anche info sulla gerarchia dei contorni)
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours


def getSrcPoints(contours, area=True):
    if area:
        # The bounding box is the one with the bigger area
        outer = max(contours, key=cv2.contourArea)
    else:
        # The bounding box is the one with the bigger perimeter
        outer = max(contours, key=lambda x: cv2.arcLength(x, True))

    #Calcolo l'epsilon per la semplificazine del poligono
    epsilon = 0.02 * cv2.arcLength(outer,True)
    #Approssimo ad un poligono
    approx = cv2.approxPolyDP(outer,epsilon,True)
    #proseguo solo se è un rettangolo
    if len(approx) == 4:
        a = approx.reshape(4, 2).astype(np.float32)
        # OuterBox middle is minimum x plus half distance from maximum x
        mid = np.min(a[:, 0]) + (np.max(a[:,0]) - np.min(a[:,0])) / 2
        # Separate left xs from right xs and sort on y
        ls = a[a[:, 0] <= mid]
        rs = a[a[:, 0] > mid]
        sort_ls = np.argsort(ls[:, 1])
        sort_rs = np.argsort(rs[:, 1])
        return np.concatenate((ls[sort_ls], rs[sort_rs]))


def getSrcPoints_threshold(contours):

    for c in contours:
        if cv2.contourArea(c) > 90000:
            #Calcolo l'epsilon per la semplificazine del poligono
            epsilon = 0.01 * cv2.arcLength(c,True)
            #Approssimo ad un poligono
            approx = cv2.approxPolyDP(c,epsilon,True)
            #proseguo solo se è un rettangolo
            if len(approx) == 4:
                a = approx.reshape(4, 2).astype(np.float32)
                # OuterBox middle is minimum x plus half distance from maximum x
                mid = np.min(a[:, 0]) + (np.max(a[:,0]) - np.min(a[:,0])) / 2
                # Separate left xs from right xs and sort on y
                ls = a[a[:, 0] <= mid]
                rs = a[a[:, 0] > mid]
                sort_ls = np.argsort(ls[:, 1])
                sort_rs = np.argsort(rs[:, 1])
                return np.concatenate((ls[sort_ls], rs[sort_rs]))


#Funzione che date le coordinate degli angoli dell'immagine ridà una vista dall'alto della stessa
def getBirdEye(img,src_points):

        #creiamo i punti di destinazione
        dst_points = np.array([
                        [0,0],
                        [0,900],
                        [900,0],
                        [900,900]
        ],dtype = np.float32)

        #computazione della matrice di omografia
        BE = cv2.getPerspectiveTransform(src_points,dst_points)

        #applicazione di Bird eye view all'immagine originale
        warped = cv2.warpPerspective(img,BE,(900,900))
        return warped

