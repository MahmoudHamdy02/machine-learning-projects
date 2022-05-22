import pandas as pd
from PIL import Image
import numpy as np
import glob
from sklearn.decomposition import PCA
import pickle
import cv2

# Define functions
def distance(x_test,x):
    return sum((x_test - x)**2)**0.5

def image_distance(images,test):
    new_array = []
    for i in range(3):
        new_array.append(distance(images[i], test))
    return round(np.mean(new_array))

# Get paths for images
train_images = glob.glob('C:/Users/Mahmoud/Desktop/VSCodeProjects/STP_ML/Train/*.jpg')
test_images = glob.glob('C:/Users/Mahmoud/Desktop/VSCodeProjects/STP_ML/test/*.jpg')

# Face recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert and crop images
train_data, test_data = [], []
for i in range(len(train_images)):
    img = Image.open(train_images[i])
    img = np.asarray(img)
    faces = face_cascade.detectMultiScale(img, 1.005, 8, minSize=(100,100))
    for (x, y, w, h) in faces:
        faces = img[y:y + h, x:x + w]
        faces = cv2.resize(faces, (180,180), interpolation=cv2.INTER_LANCZOS4)
        # cv2.imshow('image', faces)
        # cv2.waitKey()
    train_data.append(faces.flatten())


for i in range(len(test_images)):
    img = Image.open(test_images[i])
    img = np.asarray(img)
    faces = face_cascade.detectMultiScale(img, 1.005, 8, minSize=(100,100))
    for (x, y, w, h) in faces:
        faces = img[y:y + h, x:x + w]
        faces = cv2.resize(faces, (180,180), interpolation=cv2.INTER_LANCZOS4)
        # cv2.imshow('image', faces)
        # cv2.waitKey()
    test_data.append(faces.flatten())

# Fit and transform data
pca_model = PCA(n_components=30)
pca_model.fit(train_data)
new_train_data = pd.DataFrame(pca_model.transform(train_data))
new_test_data = pd.DataFrame(pca_model.transform(test_data))

# Calculate accuracy
print(np.cumsum(pca_model.explained_variance_ratio_)[-1])

# Finding distances for calculating threshold
correct_results, wrong_results = [], []
for i in range(0, 11):
    correct_results.append(image_distance([new_train_data.loc[i*3], new_train_data.loc[(i*3)+1], new_train_data.loc[(i*3)+2]], new_test_data.loc[i]))
for i in range(11, 15):
    for j in range(11):
        wrong_results.append(image_distance([new_train_data.loc[j*3], new_train_data.loc[(j*3)+1], new_train_data.loc[(j*3)+2]], new_test_data.loc[i]))

# print(correct_results, max(correct_results))
# print(wrong_results, min(wrong_results))

threshold = (max(correct_results)+min(wrong_results))/2
print(threshold)

# Export model
pickle.dump(pca_model, open('final.sav', 'wb'))