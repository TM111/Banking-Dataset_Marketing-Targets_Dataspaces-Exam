import functions as F
import pandas as pd
import random as rnd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

small=0
lenght=300
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

dataset=F.categoricalToNumeric(dataset) # no/yes -> 0/2

#TRAIN e TEST
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(dataset.columns)-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#FEATURES SCALING
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#TRAINING and PREDICTIONS
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#EVALUATING
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))