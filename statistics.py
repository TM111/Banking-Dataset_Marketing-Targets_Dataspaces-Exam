import functions as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


small=0
lenght=20
random=0
#PREPARAZIONE DATASET
dataset_path = "D:/Desktop/dataspaces/bank_full.csv"
dataset=F.formatDataset(pd.read_csv(dataset_path, delimiter=","),small,lenght,random)
for i in range(0,len(dataset)):
    F.removeAttributeValue(dataset,i, "unknown",small)

#MATRICE DI CORRELAZIONE
dataset_df = pd.DataFrame({
    'age':F.getColumn(train, "age"),
    'balance':F.getColumn(train, "balance"),
    'duration': F.getColumn(train, "duration"),
    'campaign': F.getColumn(train, "campaign"),
    'pdays': F.getColumn(train, "pdays"),
    'previous': F.getColumn(train, "previous"),
    'Class': F.getColumn(train, "Class")
})

corr_df = dataset_df.corr(method='pearson')
plt.figure(figsize=(32, 24))
sns.set(font_scale=3.5)
#sns.heatmap(corr_df, annot=True)

#MATRICE DI DISPERSIONE
plt.figure(figsize=(32, 24))
sns.set(font_scale=1.4)
#sns.pairplot(train_df)

#STATISTIC
data=[]
columns=list(dataset_df.columns)[:-1]
plt.figure(figsize=(35, 20))
for att in columns:
    data.append(list(F.getStatistic(train,att).values()))
#F.drawTable(data, columns, list(F.getStatistic(train,"age").keys()))



#BOXPLOT  (cambiare y per ogni attributo)
flierprops = dict(markerfacecolor='0.5', markersize=18,linestyle='none')
sns.set(font_scale=4)
#sns.boxplot(x="class", y="duration", linewidth=4,data=train_df, flierprops=flierprops)

#BAR CHARTS    (variabili qualitative)
attributes=["job","marital","education","default","housing","loan","contact","month","poutcome"]

att=attributes[8]
wid=0.5
plt.figure(figsize=(60, 24))

keys,no_list,yes_list=F.getOccurrences(train,att,0,1)
plt.bar(keys, no_list,width=wid)
plt.bar(keys, yes_list, bottom = no_list,width=wid)
plt.show()







