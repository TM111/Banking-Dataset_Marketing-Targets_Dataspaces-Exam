import functions as F
import pandas as pd
import random as rnd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

small=0
lenght=10000
random=1
#PREPARAZIONE DATASET
dataset_path = "D:/Desktop/dataspaces/bank_full.csv"
if(small==1):
    if(random==1):
        n = sum(1 for line in open(dataset_path)) - 1
        skip = sorted(rnd.sample(range(1,n+1),n-lenght)) 
        dataset = pd.read_csv(dataset_path,delimiter=";", skiprows=skip)
    else:
        dataset=pd.read_csv(dataset_path,delimiter=";", nrows=lenght)
else:
    dataset=pd.read_csv(dataset_path,delimiter=";")

dataset.drop(["job","marital","education","poutcome","duration","month","day_of_week",
              "default","pdays"], axis=1, inplace=True) # elimino attributi
indexRows=[]
index=-1
for row in dataset.iloc:
    index=index+1
    if("unknown" in row.values):
        indexRows.append(index)
dataset.drop(indexRows , inplace=True) #elimino missing values


for i in range(2,rnd.randint(3,6)):#mescolo il dataset
    for j in range(2,rnd.randint(3,6)):
        if(random==1):
            dataset = dataset.sample(frac=1).reset_index(drop=True) 
print("encoding...")
dataset=F.categoricalToNumeric(dataset) # no/yes -> 0/2

#TRAIN, VAL e TEST
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(dataset.columns)-1].values

test_ratio = 0.10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
'''
scaler = StandardScaler() #scaling
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
 '''
#TUNING
val_ratio = 0.15
kf = KFold(n_splits=int((1-test_ratio)/val_ratio))
kf.get_n_splits(X_train)
accuracies = []
K_values=[]
best_accuracy=0;
best_k=0
max_K=60
min_K=3
for i in range(min_K, max_K):
    knn = KNeighborsClassifier(n_neighbors=i)
    mean_accuracies=[]
    for train_index, test_index in kf.split(X_train):# KFold Cross Val
        X_train_tmp, X_val = X_train[train_index], X_train[test_index]
        y_train_tmp, y_val = y_train[train_index], y_train[test_index]
        knn.fit(X_train_tmp, y_train_tmp)
        pred_i = knn.predict(X_val)
        mean_accuracies.append(1-np.mean(pred_i != y_val))
    acc=sum(mean_accuracies)/len(mean_accuracies)
    accuracies.append(acc)
    if(best_accuracy<acc):
        best_accuracy=acc
        best_k=i
        
plt.figure(figsize=(12, 6))
plt.plot(range(min_K, max_K), accuracies, color='blue')
plt.title('Accuracy K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
