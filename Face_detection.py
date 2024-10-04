import cv2 as cv 
import numpy as np

# Load image
img = cv.imread('C:/Users/rajas/OneDrive/Desktop/Photoes/seven.jpg')  # Change backslashes to forward slashes
cv.imshow('Valantino', img)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Load Haar Cascade Classifier
haar_cascade = cv.CascadeClassifier('C:/Users/rajas/Opencv_project/FACE_PROJECT/haarcascade_classifier__face.xml')  # Path should be a string

# Detect faces
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)  # Corrected minNeighbours to minNeighbors
print(f'Number of faces found = {len(face_rect)}')

# Draw rectangles around detected faces
for (x, y, w, h) in face_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img) # face needed to be perpendicular to the camera to be detected !!!!

cv.waitKey(0)
cv.destroyAllWindows()
