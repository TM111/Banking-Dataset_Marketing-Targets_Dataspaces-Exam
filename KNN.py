import functions as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

small=1
lenght=20
random=1
#PREPARAZIONE TRAIN
train_path = "D:/Desktop/dataspaces/train.csv"
train=F.formatDataset(pd.read_csv(train_path, delimiter=","),small,lenght,random)
for i in range(0,15):
    F.removeAttributeValue(train,i, "unknown",small)
F.removeAttribute(train,i, "unknown",small)
train=F.categoricalToNumeric(train)
print(train[0])
train = DataFrame(train,columns=F.getAttributes())
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, 16].values

#PREPARAZIONE TEST
test_path = "D:/Desktop/dataspaces/test.csv"
test=F.formatDataset(pd.read_csv(test_path, delimiter=","),small,lenght,random)
for i in range(0,15):
    F.removeAttributeValue(test,i, "unknown",small)
test = DataFrame(test,columns=F.getAttributes())
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, 16].values





