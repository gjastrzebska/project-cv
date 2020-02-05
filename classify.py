import numpy as np
from PIL import  Image
import os, cv2

#Metoda służąca do wytrenowania klasyfikatora do rozpoznawania twarzy
def train_classifer(data_dir):
    #Wczytywanie wszystkich zdjęć ze zbioru
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    #Przechowywanie obrazu w formacie numpy i ID użytkownika o tym samym indeksie na listach
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    #Trenowanie oraz zapisywanie klasyfikatora
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")


train_classifer("data")