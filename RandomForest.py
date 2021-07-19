import functions as F
import pandas as pd
import random as rnd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import sys
from mlxtend.plotting import plot_learning_curves
from sklearn.ensemble import RandomForestClassifier


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


#TUNING
val_ratio = 0.15
kf = KFold(n_splits=int((1-test_ratio)/val_ratio))
kf.get_n_splits(X_train)

accuracies = []
Criterions=["gini","entropy"]
Max_depth=[]
for i in range(2,40):
    Max_depth.append(i)
min_samples_splits=[]
for i in range(2,50):
    min_samples_splits.append(i)
n_estimators=[]
for i in range(1,11):
    min_samples_splits.append(i*50)
    
best_accuracy=0;
best_C=0
best_D=0
best_MSS=0
best_N=0
i=-1

if(small):
    Max_depth=[8]
    min_samples_splits=[5,15]
    n_estimators=[100,200]
    
max_i=len(Criterions)*len(Max_depth)*len(min_samples_splits)*len(n_estimators)*int((1-test_ratio)/val_ratio)

for c in Criterions:
    for d in Max_depth:
        for mss in min_samples_splits:
            for num in n_estimators:
                clf = RandomForestClassifier(n_estimators=num,criterion=c,max_depth=d,max_features="sqrt",min_samples_split=mss,random_state=0)
                mean_accuracies=[]
                for train_index, test_index in kf.split(X_train):# KFold Cross Val
                    i=i+1
                    percentage="Tuning: "+str(int(100*(i)/max_i))+"%"
                    sys.stdout.write('\r'+percentage)
        
                    X_train_tmp, X_val = X_train[train_index], X_train[test_index]
                    y_train_tmp, y_val = y_train[train_index], y_train[test_index]
                    clf = clf.fit(X_train_tmp, y_train_tmp)
                    pred_i = clf.predict(X_val)
                    mean_accuracies.append(1-np.mean(pred_i != y_val))
                acc=sum(mean_accuracies)/len(mean_accuracies)
                accuracies.append(acc)
                if(best_accuracy<acc):
                    best_accuracy=acc
                    best_C=c
                    best_D=d
                    best_MSS=mss
                    best_N=num
sys.stdout.write('\r'+"                                             "+'\r')


#TEST
clf = RandomForestClassifier(n_estimators=best_N,criterion=best_C,max_depth=best_D,max_features="sqrt",min_samples_split=best_MSS,random_state=0)
clf.fit(X_train, y_train)
pred_i = clf.predict(X_test)
acc=1-np.mean(pred_i != y_test)
print('\r'+"Criterion: ",best_C)
print('\r'+"Max_depth: ",best_D)
print('\r'+"MSS: ",best_MSS)
print('\r'+"N estimators: ",best_N)
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



#probability=True
clf = RandomForestClassifier(n_estimators=best_N,criterion=best_C,max_depth=best_D,max_features="sqrt",min_samples_split=best_MSS,random_state=0,)
clf.fit(X_train, y_train)

y_scores = clf.predict_proba(X_test)
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
clf = RandomForestClassifier(n_estimators=best_N,criterion=best_C,max_depth=best_D,max_features="sqrt",min_samples_split=best_MSS,random_state=0)
clf.fit(X_train, y_train)
plot_learning_curves(X_train, y_train, X_test, y_test, clf)
plt.show()
