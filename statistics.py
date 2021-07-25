import functions as F
import seaborn as sns
import matplotlib.pyplot as plt


#LOAD E PREPROCESSIONG DATASET
small=1
lenght=100
random=1

dataset=F.getDataset(small,lenght,random)


#MATRICE DI CORRELAZIONE
attributes=["age","campaign","pdays","previous","emp.var.rate",
            "cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]

df=dataset[attributes]

corr_df = df.corr(method='pearson')
plt.figure(figsize=(50, 30))
sns.set(font_scale=3.5)
plot=0
if(plot):
    sns.heatmap(corr_df, annot=True)





#MATRICE DI DISPERSIONE
plt.figure(figsize=(44, 30))
sns.set(font_scale=1.7)
plot=0
if(plot):
    sns.pairplot(df)





#STATISTIC
data=[]
plt.figure(figsize=(45, 28))

for att in df.columns:
    data.append(list(F.getStatistic(df,att).values()))

plot=0
if(plot):
    F.drawStatisticsTable(data, list(df.columns), list(F.getStatistic(df,"age").keys()))


#BOXPLOT  (cambiare att per ogni attributo)
att="nr.employed"
flierprops = dict(markerfacecolor='0.5', markersize=18,linestyle='none')
sns.set(font_scale=4)
plot=0
plt.figure(figsize=(15,15))
if(plot):
    sns.boxplot(x="y", y=att, linewidth=4,data=dataset, flierprops=flierprops,
                hue="y")
    plt.legend(loc='upper center')
    plt.xlabel("")




#BAR CHARTS    (variabili qualitative)
att="y"
#plt.figure(figsize=(18,20))
fontsize=39
margin=0.09
height=40
total = len(dataset)

plot=0
if(plot):
    ax=sns.countplot(x=att, data=dataset, hue="y")
    ax.set(ylabel='#samples')
    plt.legend(loc='upper right')
    for p in ax.patches:
        ax.annotate('{:.2f}%'.format(100*p.get_height()/total), (p.get_x()+margin, p.get_height()+height)
                    ,fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                       fontsize=fontsize+22)
    plt.show()



#DENSITY
att="age"

#plt.xlim(-1, 5)
plot=0
if(plot):
    ax=sns.kdeplot(data=dataset, x=att, hue="y",multiple="stack")
    ax.yaxis.tick_right()
if(plot and 1==1): #move legend left
    ax.legend_.set_bbox_to_anchor((0.05, 0.95))
    ax.legend_._set_loc(2)
    plt.show()

