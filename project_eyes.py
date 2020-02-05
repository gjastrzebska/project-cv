import cv2
import sys

#klasyfikatory Cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eyeRCascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
eyeLCascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')
#noseCascade = cv2.CascadeClassifier('Nariz.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
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
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyesR = eyeRCascade.detectMultiScale(roi_gray, 1.2, 50)
        eyesL = eyeLCascade.detectMultiScale(roi_gray, 1.2, 50)
        smile = smileCascade.detectMultiScale(roi_gray, 1.1, 50)
        #nose = noseCascade.detectMultiScale(roi_gray, 1.1, 4)
        for (ex,ey,ew,eh) in eyesR:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(roi_color, 'Prawe', (ex, ey-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
        for (lx,ly,lw,lh) in eyesL:
            cv2.rectangle(roi_color,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)
            cv2.putText(roi_color, 'Lewe', (lx, ly-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 2)
            cv2.putText(roi_color, 'Usta', (sx, sy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
        #for (nx,ny,nw,nh) in nose:
           # cv2.rectangle(roi_color, (nx,ny), (nx+nw,ny+nh), (0,0,0), 2)
            #cv2.putText(roi_color, 'Nos', (nx, ny-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

    #Utworzenie okna z rezultatem
    cv2.imshow('Wykrywanie twarzy w czasie rzeczywistym', frame)

    #Aby zamknac nacisnac przycisk 'q' - quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Zamkniecie okna
video_capture.release()
cv2.destroyAllWindows()