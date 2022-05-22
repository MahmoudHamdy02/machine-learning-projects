import pandas as pd
from PIL import Image
import numpy as np
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error

def distance(x_test,x):
    return sum((x_test - x)**2)**0.5

train_images = glob.glob('C:/Users/Mahmoud/Desktop/VSCodeProjects/STP_ML/Train/*.jpg')
test_images = glob.glob('C:/Users/Mahmoud/Desktop/VSCodeProjects/STP_ML/test/*.jpg')

image1 = Image.open(train_images[0])
image2 = Image.open(train_images[1])
image3 = Image.open(train_images[2])
test1 = Image.open(test_images[9])

n = 3 # no. of pictures per person
target = []
for i in range(1, (len(train_images) // n) + 1):
    target += [i] * n
# print(target)

train_data, test_data = [], []
for i in range(len(train_images)):
    # with open(train_images[i], 'rb') as file:
        img = Image.open(train_images[i])
        # print(img)
        numpy_data = np.asarray(img)
        train_data.append(numpy_data.flatten())
for i in range(len(test_images)):
    with open(test_images[i], 'rb') as file:
        img = Image.open(file)
        numpy_data = np.asarray(img)
        test_data.append(numpy_data.flatten())
train_data = np.array(train_data)
# train_data = pd.DataFrame(train_data)
test_data = np.array(test_data)
# test_data = pd.DataFrame(test_data)

pca_model = PCA(n_components=15)
pca_model.fit(train_data)
new_data = pd.DataFrame(pca_model.transform(train_data))
new_data_test = pd.DataFrame(pca_model.transform(test_data))

# print(np.cumsum(pca_model.explained_variance_ratio_)[-1])

# new_data["target"] = target
# new_data.plot(kind="scatter", x=0,y=2,color=new_data['target'],colormap="Accent")
# plt.show()

# filename = 'final.sav'
# pickle.dump(pca_model, open(filename, 'wb'))

new_array, correct_results, wrong_results = [], [], []
for i in range(len(test_images)-4):
    new_array = []
    for j in range(3):
        new_array.append(distance(new_data.loc[j+i*3], new_data_test.loc[i]))
    correct_results.append(round(np.mean(new_array)))

for i in range(len(test_images)-4, len(test_images)):
    new_array = []
    for j in range(3):
        new_array.append(distance(new_data.loc[j], new_data_test.loc[i]))
    # print(new_array)
    wrong_results.append(round(np.mean(new_array)))

def image_distance(images,test):
    new_array = []
    for i in range(3):
        new_array.append(distance(images[i], test))
    return np.mean(new_array)

# print(correct_results, "\n")
# print(wrong_results, "\n")

threshold = (max(correct_results)+min(wrong_results))/2 # 12000


# print(image_distance([new_data.loc[0], new_data.loc[1], new_data.loc[2]], new_data_test.loc[0]))
# print([new_data.loc[0], new_data.loc[1], new_data.loc[2]], new_data_test.loc[0])

def predict(test, images):
    test_data = np.asarray(test).flatten()
    images_data = []
    for i in range(3):
        numpy_data = np.asarray(images[i])
        images_data.append(numpy_data.flatten())
    # print(images_data)
    new_images = pca_model.transform(images_data)
    new_test = pca_model.transform(test_data.reshape(1,-1))
    # print(pd.DataFrame(new_images), pd.DataFrame(new_test))
    print(image_distance(new_images, new_test))
    if image_distance(new_images, new_test) < threshold:
        return "Match! - 1"
    else:
        return "No Match! - 0"

# print(predict(pd.DataFrame([train_data.loc[0],train_data.loc[1],train_data.loc[2]]),test_data.loc[14]))
# print(pd.DataFrame([train_data.loc[0],train_data.loc[1],train_data.loc[2]]))
print(predict(test1, [image1,image2,image3]))
