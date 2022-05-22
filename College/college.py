import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("C:\\Users\\Mahmoud\Desktop\\VSCodeProjects\\STP_ML\\College\\Admission_Predict_Ver1.1.csv").drop("Serial No.", axis=1)
print(data)
data["Chance of Admit "].loc[data["Chance of Admit "]>=0.5]=1
data["Chance of Admit "].loc[data["Chance of Admit "]<0.5]=0
x = data.drop("Chance of Admit ", axis=1)
y = data["Chance of Admit "]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.1)
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

filename = 'final.sav'
# pickle.dump(model, open(filename, 'wb'))

def predict(test):
    loaded_model = pickle.load(open(filename, 'rb'))
    
    return loaded_model.predict(np.array(test)[:,np.newaxis].T)

print(predict(x_test.loc[0].values))
print(x_test.loc[0].values)