import cv2
import sys

#klasyfikatory Cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

video_capture = cv2.VideoCapture(0)

while True:
    #Przechwytywanie obrazu klatka po klatce
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=10,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h) ,(255,0,0),2)
        cv2.putText(frame, 'Twarz', (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1, cv2.LINE_AA)

    #Utworzenie okna z rezultatem
    cv2.imshow('Wykrywanie twarzy w czasie rzeczywistym', frame)

    #Aby zamknac nacisnac przycisk 'q' - quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Zamkniecie okna
video_capture.release()
cv2.destroyAllWindows()