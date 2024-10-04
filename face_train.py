import os
import cv2 as cv
import numpy as np

people = ['jk', 'rm', 'yoon']
DIR = r'C:/Users/rajas/OneDrive/Desktop/face train model'

# Ensure haarcascade file path is correct and the file exists
haar_cascade = cv.CascadeClassifier(r'C:/Users/rajas/Opencv_project/FACE_PROJECT/haarcascade_classifier__face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            # Read the image
            img_array = cv.imread(img_path)

            if img_array is None:
                continue  # Skip if the image is not read properly

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done.')

# Convert lists to NumPy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Corrected recognizer creation function
face_recogniser = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the feature list and label list
face_recogniser.train(features, labels)

# Save the trained model
face_recogniser.save(r'C:/Users/rajas/OneDrive/Desktop/face_train_model.yml')
print('YML file saved successfully.')


# Save the features and labels for future use
np.save('features.npy', features)
np.save('labels.npy', labels)
     

             
             

