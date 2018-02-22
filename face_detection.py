import cv2
import numpy as np
import os

def detect_faces(data_img, detector, scale_factor=1.1):
    img = np.copy(data_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return img

path = 'Faces_dataset'
haar_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for file1 in os.listdir(path):
    im = cv2.imread(path+"\\"+file1)
    result=detect_faces(im, haar_detector, 1.3)
    #Play around with the scale_factor for different images as the some faces might be closer to camera than the others
    cv2.imshow('img',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
