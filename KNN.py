import functions as F
import pandas as pd
import random as rnd
from sklearn.model_selection import train_test_split

small=0
lenght=500
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

dataset.drop(["duration","month","day_of_week",
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
                   dataset = dataset.sample(frac=1).reset_index(drop=True) 

train, test = train_test_split(dataset, test_size=0.1)
print("tot: ",len(dataset))
print("train: ",len(train))
print("test: ",len(test))