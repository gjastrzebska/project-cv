import cv2

#Metoda generująca zbiór danych w celu rozpoznania użytkownika
def generate_dataset(img, id, img_id):
    #zapisywanie zdjęć do folderu 'data' o nazwie 'userID.imgID.jpg
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

#Metoda rysująca prostokąt wokół wykrytych obiektów
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #wykrywanie obiektów ze zwracaniem współrzędnych, wysokosci i szerokosci
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

#Metoda wykrywająca twarz
def detect(img, faceCascade, img_id):
    color = {"blue":(255,0,0)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    #jeśli obiekt został wykryty zmienna coords będzie miała 4 dane
    if len(coords)==4:
        #aktualizowanie obiektu poprzez przycięcie obrazu do prostokąta
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_id = 1
        generate_dataset(roi_img, user_id, img_id)

    return img

#Klasyfikator
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

#Rozpoczecie numerowania zmiennej img_id od 0
img_id = 0

while True:
    if img_id % 50 == 0:
        print("Zebrano ", img_id," zdjęć")
    _, img = video_capture.read()
    #wywołanie metody detect
    img = detect(img, faceCascade, img_id)
    #zapisywanie obrazu w nowym oknie
    cv2.imshow("Wykrywanie twarzy", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()