import cv2


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        #Przewidywanie ID użytkownika
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        #Sprawdzanie ID użytkownika i przypisanie mu etykiety
        if id==1:
            cv2.putText(img, "Gabriela", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

#Metoda do rozpoznawania osoby
def recognize(img, clf, faceCascade):
    color = {"white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.2, 20, color["white"], "Twarz", clf)
    return img


#Klasyfikator
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Wczytywanie zapisanego klasyfikatora do rozpoznawania
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    #wywołanie metody
    img = recognize(img, clf, faceCascade)
    #Otwieranie okna wraz z rozpoznawaniem użytkownika
    cv2.imshow("Rozpoznawanie", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()