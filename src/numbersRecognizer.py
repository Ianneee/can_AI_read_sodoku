import cv2
import torch
import numpy as np
from torchvision import transforms
from digitsCnn import DigitsCNN

IMG_SIZE = 28

class NumbersRecognizer():
    def __init__(self):
        self.cleaned_data = [];
        self.predictions = [];
        self.model = self.load_torch_model()

    def load_torch_model(self):
        model = DigitsCNN()
        model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
        return model

    def mask(self):
        mask = np.full((IMG_SIZE, IMG_SIZE), 255).astype(np.uint8)

        mask[:, :4] = 0
        mask[:, IMG_SIZE-4:] = 0
        mask[:4, :] = 0
        mask[IMG_SIZE-4:, :] = 0

        return mask

    def clean_images(self, quads):
        # Delete borders of the grid
        mask = self.mask()

        for q in quads:
            resized = cv2.resize(q, (IMG_SIZE, IMG_SIZE))

            # Sharp immag
            #smoothed = cv2.GaussianBlur(resized, (9,9), 10)
            #resized = cv2.addWeighted(resized, 1.5, smoothed, -0.5, 0)

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            #edges = cv2.Canny(gray,50,100)
            edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # Remove borders from image
            masked = cv2.bitwise_and(edges, edges, mask=mask)

            n = np.zeros(masked.shape, dtype="uint8")
            blank = np.zeros(masked.shape, dtype="uint8")
            if np.sum(masked) != 0:

                contours, _ = cv2.findContours(masked,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cont = max(contours, key=cv2.contourArea)

                cv2.drawContours(blank, [cont], -1, (255, 255, 255), thickness=cv2.FILLED)

                n = cv2.bitwise_and(blank, blank, masked)
            self.cleaned_data.append(n)

    def recognize(self, sudoku_cells):
        self.cleaned_data = [];
        self.predictions = []
        self.clean_images(sudoku_cells)
        for q in self.cleaned_data:
            if np.sum(q) != 0:
                tf = transforms.Compose([transforms.ToTensor()])
                img = tf(q).float().unsqueeze(0)
                pred = self.model(img)
                self.predictions.append(pred.data.numpy().argmax())

            else:
                self.predictions.append(None)

