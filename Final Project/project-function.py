import pandas as pd
import numpy as np
from PIL import Image
import cv2
import glob
import pickle

# Import model
pca_model = pickle.load(open('final.sav', 'rb'))

# Face recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get paths for testing
train_images = glob.glob('C:/Users/Mahmoud/Desktop/VSCodeProjects/STP_ML/Train/*.jpg')
test_images = glob.glob('C:/Users/Mahmoud/Desktop/VSCodeProjects/STP_ML/test/*.jpg')
test_new_images = glob.glob('C:/Users/Mahmoud/Desktop/VSCodeProjects/STP_ML/test_new/*.jpg')

# Setting test images
image1 = Image.open('C:/Users/Mahmoud/Desktop/Photos/image4-1.jpeg')
image2 = Image.open('C:/Users/Mahmoud/Desktop/Photos/image4-2.jpeg')
image3 = Image.open('C:/Users/Mahmoud/Desktop/Photos/image4-3.jpeg')
test1 = Image.open('C:/Users/Mahmoud/Desktop/Photos/two.jpg')

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
            # Uncomment to see the cropped faces
            cv2.imshow('image', faces)
            cv2.waitKey()
        images_list.append(faces.flatten())
    images_list = pd.DataFrame(pca_model.transform(images_list))

    test_data = np.asarray(test)
    if test.mode != "L":
        test_data = cv2.cvtColor(test_data, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(test_data, 1.05, 8, minSize=(300,300))
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            faces = test_data[y:y + h, x:x + w]
            faces = cv2.resize(faces, (180,180), interpolation=cv2.INTER_LANCZOS4)
            cv2.imshow('image', faces)
            cv2.waitKey()
        test = faces.flatten()
        test = pca_model.transform([test])
        print(image_distance([images_list.loc[0],images_list.loc[1],images_list.loc[2]], test))
        if image_distance([images_list.loc[0],images_list.loc[1],images_list.loc[2]], test) < 4300:
            return "Match!"
        else:
            return "No match"
    else:
        return "No match"

print(predict(test1, [image1,image2,image3]))

results = []
for i in range(11, 15):
    for j in range(11):
        results.append(predict(Image.open(test_images[i]), [Image.open(train_images[j*3]), Image.open(train_images[j*3]), Image.open(train_images[j*3])]))
print(results)
# Tests for different images
for i in range(len(test_new_images)):
    results.append(predict(Image.open(test_new_images[i]), [Image.open(train_images[(i//10)*3]), Image.open(train_images[(i//10)*3+1]), Image.open(train_images[(i//10)*3+2])]))
print(results)
counter = 0
for i in range(len(results)):
    if results[i] == "No match":
        counter += 1
        print(test_new_images[i])
print(110-counter)
