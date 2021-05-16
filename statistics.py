import functions as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd


small=1
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
        if(random==1):
            dataset = dataset.sample(frac=1).reset_index(drop=True) 



#MATRICE DI CORRELAZIONE

df=dataset[["age","campaign","previous","emp.var.rate",
            "cons.price.idx","cons.conf.idx","euribor3m"]]

corr_df = df.corr(method='pearson')
plt.figure(figsize=(32, 24))
sns.set(font_scale=3.5)
#sns.heatmap(corr_df, annot=True)

#MATRICE DI DISPERSIONE
plt.figure(figsize=(32, 24))
sns.set(font_scale=1.4)
#sns.pairplot(df)

#STATISTIC
data=[]
plt.figure(figsize=(35, 20))
for att in df.columns:
    data.append(list(F.getStatistic(df,att).values()))

#F.drawTable(data, list(df.columns), list(F.getStatistic(df,"age").keys()))


#BOXPLOT  (cambiare y per ogni attributo)
attributes=["age","campaign","previous","emp.var.rate",
            "cons.price.idx","cons.conf.idx","euribor3m"]
att=attributes[0]
flierprops = dict(markerfacecolor='0.5', markersize=18,linestyle='none')
sns.set(font_scale=4)
#sns.boxplot(x="y", y=att, linewidth=4,data=dataset, flierprops=flierprops)

#BAR CHARTS    (variabili qualitative)
attributes=["job","marital","education","housing",
            "loan","contact","poutcome"]

att=attributes[2]
wid=0.5
plt.figure(figsize=(60, 24))

keys,no_list,yes_list=F.getOccurrences(dataset,att,1,1)
plt.bar(keys, no_list,width=wid)
plt.bar(keys, yes_list, bottom = no_list,width=wid)
plt.show()






