# pylint:disable=no-member

import numpy as np
import cv2 as cv

# Load Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('C:/Users/rajas/Opencv_project/FACE_PROJECT/haarcascade_classifier__face.xml')

# List of people corresponding to label indices
people = ['jk', 'rm', 'yoon']

# Load the trained face recognizer model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/rajas/Opencv_project/FACE_PROJECT/face_train_model.yml')

# Read the image
img = cv.imread(r'C:/Users/rajas/OneDrive\Desktop/face train model/yoon/1.jpg')  # Correct the image file extension if needed

# Convert image to grayscale for face detection and recognition
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Loop over all detected faces
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]  # Extract the region of interest (ROI) where the face is detected

    # Perform face recognition on the detected face
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # Display the label and rectangle around the face on the original image
    cv.putText(img, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

# Show the image with detected and recognized face
cv.imshow('Detected Face', img)

# Wait for a key press before closing the windows
cv.waitKey(0)
cv.destroyAllWindows()
