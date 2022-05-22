import pandas as pd
import numpy as np
import cv2
import pickle

# Import model
pca_model = pickle.load(open('final.sav', 'rb'))

# Face recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Define functions
def distance(x_test,x):
    return sum((x_test - x)** 2) **0.5

def image_distance(images,test):
    new_array = []
    for i in range(3):
        new_array.append(distance(images[i], test[0]))
    return round(np.mean(new_array))

# Images is a list of 3 PIL Image objects
def predict(test, images):
    images_list = []
    for i in range(3):
        numpy_data = np.asarray(images[i])
        if images[i].mode != "L":
            numpy_data = cv2.cvtColor(numpy_data, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(numpy_data, 1.05, 8, minSize=(300,300))
        for (x, y, w, h) in faces:
            faces = numpy_data[y:y + h, x:x + w]
            faces = cv2.resize(faces, (180,180), interpolation=cv2.INTER_LANCZOS4)
            # cv2.imshow('image', faces)
            # cv2.waitKey()
        images_list.append(faces.flatten())
    images_list = pd.DataFrame(pca_model.transform(images_list))

    test_data = np.asarray(test)
    if test.mode != "L":
        test_data = cv2.cvtColor(test_data, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(test_data, 1.05, 8, minSize=(300,300))
    for (x, y, w, h) in faces:
        faces = test_data[y:y + h, x:x + w]
        faces = cv2.resize(faces, (180,180), interpolation=cv2.INTER_LANCZOS4)
        # cv2.imshow('image', faces)
        # cv2.waitKey()
    if len(faces) != 0:
        test = faces.flatten()
        test = pca_model.transform([test])
        # print(image_distance([images_list.loc[0],images_list.loc[1],images_list.loc[2]], test))
        if image_distance([images_list.loc[0],images_list.loc[1],images_list.loc[2]], test) < 5500:
            return 1 #Match!
        else:
            return 0 #No match...
    else:
        return 0 #No match...