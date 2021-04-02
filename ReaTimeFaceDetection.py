# OpenCV Real Time Face Detection!

# import numpy as np
#module OpenCV
import cv2 
cam = cv2.VideoCapture(0)
# Source File : https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        Cface = [(w/2+x),(h/2+y)]
        print (str(Cface[0]) + "," + str(Cface[1])) #Error Jika tidak dilengkapi ()
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey), (ex + ew, ey + eh), (0, 255, 0 ), 2)
    
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cam.release()
cv2.destroyAllWindows() 
