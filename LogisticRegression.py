import functions as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import sys
from mlxtend.plotting import plot_learning_curves
from sklearn.linear_model import LogisticRegression


#LOAD E PREPROCESSIONG DATASET
small=1
lenght=1000
random=1

dataset=F.getDataset(small,lenght,random)

#ENCODING
dataset=F.encodingDataset(dataset)


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
C_values=[100, 10, 1.0, 0.1, 0.01]
Penalties=["l2","l1"]

if(small):
    Penalties=["l2"]

best_accuracy=0;
best_C=0
best_p=0
i=-1

mi=10000 #max_iter
max_i=len(C_values)*len(Penalties)*int((1-test_ratio)/val_ratio)

for p in Penalties:
    for c in C_values:
        clf = LogisticRegression(penalty=p,C=c,solver="liblinear",random_state=0,max_iter=mi)
        mean_accuracies=[]
        for train_index, test_index in kf.split(X_train):# KFold Cross Val
            i=i+1
            percentage="Tuning: "+str(int(100*(i)/max_i))+"%"
            sys.stdout.write('\r'+percentage)
        
            X_train_tmp, X_val = X_train[train_index], X_train[test_index]
            y_train_tmp, y_val = y_train[train_index], y_train[test_index]
            clf.fit(X_train_tmp, y_train_tmp)
            pred_i = clf.predict(X_val)
            mean_accuracies.append(1-np.mean(pred_i != y_val))
        acc=sum(mean_accuracies)/len(mean_accuracies)
        accuracies.append(acc)
        if(best_accuracy<acc):
            best_accuracy=acc
            best_C=c
            best_p=p
sys.stdout.write('\r'+"                                             "+'\r')

#TEST
clf = LogisticRegression(penalty=best_p,C=best_C,solver="liblinear",random_state=0,max_iter=mi)
clf.fit(X_train, y_train)
pred_i = clf.predict(X_test)
acc=1-np.mean(pred_i != y_test)
print('\r'+"C: ",best_C)
print('\r'+"Penalty: ",best_p)
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


clf = LogisticRegression(penalty=best_p,C=best_C,solver="liblinear",random_state=0,max_iter=mi)
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
clf = LogisticRegression(penalty=best_p,C=best_C,solver="liblinear",random_state=0,max_iter=mi)
clf.fit(X_train, y_train)
plot_learning_curves(X_train, y_train, X_test, y_test, clf)
plt.show()