import functions as F
import pandas as pd
import random as rnd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import sys
from sklearn.svm import SVC
from mlxtend.plotting import plot_learning_curves

small=1
lenght=1000
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

#dataset.drop(["job","marital","education","poutcome","default"], axis=1, inplace=True) # elimino attributi
dataset.drop(["duration","month","day_of_week",
              "pdays"], axis=1, inplace=True) # elimino attributi

dataset=F.deleteMissingValues(dataset, "unknown")

for i in range(2,rnd.randint(3,6)):#mescolo il dataset
    for j in range(2,rnd.randint(3,6)):
        if(random==1):
            dataset = dataset.sample(frac=1).reset_index(drop=True) 

#dataset=F.labelEncoder(dataset,["job"])
dataset=F.OneHotEncoder(dataset,["housing","loan","contact","job","marital","education","poutcome","default"]) 

#TRAIN, VAL e TEST
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(dataset.columns)-1].values

test_ratio = 0.10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)

scaler = StandardScaler() #scaling
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#TUNING
val_ratio = 0.15
kf = KFold(n_splits=int((1-test_ratio)/val_ratio))
kf.get_n_splits(X_train)


accuracies = []
C_values=[]
for i in range(-2,4):
    C_values.append(pow(10,i))
Gamma_values=[]
for i in range(-4,0):
    Gamma_values.append(pow(10,i))

best_accuracy=0;
best_C=0
best_g=0
i=-1
max_i=len(C_values)*len(Gamma_values)*int((1-test_ratio)/val_ratio)

for c in C_values:
    for g in Gamma_values:
        svm = SVC(C=c, gamma=g)
        mean_accuracies=[]
        for train_index, test_index in kf.split(X_train):# KFold Cross Val
            i=i+1
            percentage="Tuning: "+str(int(100*(i)/max_i))+"%"
            sys.stdout.write('\r'+percentage)
        
            X_train_tmp, X_val = X_train[train_index], X_train[test_index]
            y_train_tmp, y_val = y_train[train_index], y_train[test_index]
            svm.fit(X_train_tmp, y_train_tmp)
            pred_i = svm.predict(X_val)
            mean_accuracies.append(1-np.mean(pred_i != y_val))
        acc=sum(mean_accuracies)/len(mean_accuracies)
        accuracies.append(acc)
        if(best_accuracy<acc):
            best_accuracy=acc
            best_C=c
            best_g=g
sys.stdout.write('\r'+"                                             "+'\r')

#TEST
svm = SVC(C=best_C, gamma=best_g)
svm.fit(X_train, y_train)
pred_i = svm.predict(X_test)
acc=1-np.mean(pred_i != y_test)
print('\r'+"C: ",best_C)
print('\r'+"Gamma: ",best_g)
print("Accuracy: ",round(acc,3))

#ROC CURVE
catToNumDict = {
   "no": 0,
  "yes": 1,
}
for i in range(0,len(y_train)):
    y_train[i]=catToNumDict[y_train[i]]

for i in range(0,len(y_test)):
    y_test[i]=catToNumDict[y_test[i]]
y_train=y_train.astype('int')
y_test=y_test.astype('int')


svm = SVC(C=best_C, gamma=best_g,probability=True)
svm.fit(X_train, y_train)


y_scores = svm.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 8))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

#LEARNING CURVE
svm = SVC(C=best_C, gamma=best_g)
svm.fit(X_train, y_train)
plot_learning_curves(X_train, y_train, X_test, y_test, svm)
plt.show()
