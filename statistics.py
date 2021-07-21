import functions as F
import seaborn as sns
import matplotlib.pyplot as plt



#LOAD E PREPROCESSIONG DATASET
small=1
lenght=1000
random=1

dataset=F.getDataset(small,lenght,random)





#MATRICE DI CORRELAZIONE
attributes=["age","campaign","previous","emp.var.rate",
            "cons.price.idx","cons.conf.idx","euribor3m"]
df=dataset[attributes]

corr_df = df.corr(method='pearson')
plt.figure(figsize=(32, 24))
sns.set(font_scale=3.5)
plot=1
if(plot):
    sns.heatmap(corr_df, annot=True)





#MATRICE DI DISPERSIONE
plt.figure(figsize=(32, 24))
sns.set(font_scale=1.4)
plot=0
if(plot):
    sns.pairplot(df)





#STATISTIC
data=[]
plt.figure(figsize=(35, 20))
for att in df.columns:
    data.append(list(F.getStatistic(df,att).values()))

plot=0
if(plot):
    F.drawTable(data, list(df.columns), list(F.getStatistic(df,"age").keys()))





#BOXPLOT  (cambiare y per ogni attributo)
att=attributes[0]
flierprops = dict(markerfacecolor='0.5', markersize=18,linestyle='none')
sns.set(font_scale=4)
plot=0
if(plot):
    sns.boxplot(x="y", y=att, linewidth=4,data=dataset, flierprops=flierprops)






#BAR CHARTS    (variabili qualitative)
attributes=["job","marital","education","default","housing",
            "loan","contact","poutcome"]
att=attributes[6]
wid=0.5
plt.figure(figsize=(60, 24))
normalized=0
order=1
keys,no_list,yes_list=F.getOccurrences(dataset,att,normalized,order)
plot=0
if(plot):
    plt.bar(keys, no_list,width=wid)
    plt.bar(keys, yes_list, bottom = no_list,width=wid)
    plt.show()






