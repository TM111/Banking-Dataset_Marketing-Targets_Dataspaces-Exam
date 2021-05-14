import functions as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

small=0
lenght=20
random=1
#PREPARAZIONE TRAIN
train_path = "D:/Desktop/dataspaces/train.csv"
train=F.formatDataset(pd.read_csv(train_path, delimiter=","),small,lenght,random)
for i in range(0,15):
    train=F.removeAttributeValue(train,i, "unknown",0)
train=F.categoricalToNumeric(train)

train = DataFrame(train,columns=F.getAttributes())
del train["job"]
del train["marital"]
del train["day"]
del train["month"]
del train["poutcome"]
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, 11].values

#PREPARAZIONE DATASET
test_path = "D:/Desktop/dataspaces/test.csv"
test=F.formatDataset(pd.read_csv(test_path, delimiter=","),small,lenght,random)
for i in range(0,15):
    test=F.removeAttributeValue(test,i, "unknown",0)
test=F.categoricalToNumeric(test)
test = DataFrame(test,columns=F.getAttributes())

del test["job"]
del test["marital"]
del test["day"]
del test["month"]
del test["poutcome"]
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, 11].values

#FEATURE SCALING
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#CLASSIFIER FIT
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
count=0
for i in y_pred:
    if(i==1):
        count=count+1
print(count)
