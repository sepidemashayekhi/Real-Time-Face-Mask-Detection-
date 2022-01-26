from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
import numpy as np

import cv2

face_classifier=cv2.CascadeClassifier(r'G:/my_project/Facial-expression-recognition/haarcascade_frontalface_default.xml')
classifier=load_model('my_bestModel.h5')
emotion_labels=['Whit Mask','Without Mask']

print('[INFO] starting video stream..')
cam = cv2.VideoCapture(0)
while True:

    _, frame = cam.read()
    face_loc = face_classifier.detectMultiScale(frame)
    for (x, y, w, h) in face_loc:
        roi_gray = frame[y:y + h, x:x + w]
        (X,Y,W,H)=(x,y,w,h)
    if len(roi_gray) != 0:
        face = cv2.resize(roi_gray, (224, 224))
        face = preprocess_input(face)
        face = np.reshape(face, (1, 224, 224, 3))
        predict = classifier.predict(face)[0]
        if predict<0.5:
            label=emotion_labels[0]
            color = (0, 255, 0)
        elif predict>=0.5:
            label = emotion_labels[1]
            color = (0, 0, 255)


    cv2.rectangle(frame,(X,Y),(X+W,Y+H),color=color,thickness=4)
    cv2.putText(frame,label,(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

    cv2.imshow('face', roi_gray)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)==27:
         break




