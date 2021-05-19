import numpy as np
import matplotlib.pyplot as plt
import sys
catToNumDict = {
   "no": 0,
  "yes": 2,
  "telephone": 0,
  "cellular": 2,
}

def categoricalToNumeric(ds):
    dataset=ds
    attributes=["housing","loan","contact"]
    for index, row in dataset.iterrows():
        percentage="encoding: "+str(int(100*index/len(ds)))+"%"
        sys.stdout.write('\r'+percentage)
        for att in attributes:
            value=str(dataset.iloc[index][att])
            dataset.loc[index, att]=catToNumDict[value]
    sys.stdout.write('\r'+"                                             "+'\r')
    return dataset

def getOccurrences(ds,attribute,normalize=0,order=0):
    column=ds[attribute]
    total_sum=len(column)
    values={}
    for v in column:
        if(v not in values):
            values[v]=[0,0]
    
    for i in range(0,len(ds)):
        value=ds.iloc[i][attribute]
        c=ds.iloc[i]["y"]
        if(c=="no" or c==0):
            values[value]=[values[value][0]+1,values[value][1]]
        else:
            values[value]=[values[value][0],values[value][1]+1]

    keys=list(values.keys())
    for i in range(0,len(keys)):
        tmp=keys[i]+"\n"+str(round((values[keys[i]][0]+values[keys[i]][1])*100/total_sum,1))+"%"
        keys[i]=tmp+"\n"+str(values[keys[i]][0]+values[keys[i]][1])
    no_list=[]
    yes_list=[]
    for k in values.keys():
        sum=values[k][0]+values[k][1]
        if(normalize==0):
            sum=1
        no_list.append(values[k][0]/sum)
        yes_list.append(values[k][1]/sum)
    if(order==1):
        if(normalize==1):
            for i in range(0,len(yes_list)-1):
                for j in range(0,len(yes_list)-1):
                    if(yes_list[j]>yes_list[j+1]):
                        yes_list[j], yes_list[j+1] = yes_list[j+1], yes_list[j]
                        no_list[j], no_list[j+1] = no_list[j+1], no_list[j]
                        keys[j], keys[j+1] = keys[j+1], keys[j]
        else:
            for i in range(0,len(yes_list)-1):
                for j in range(0,len(yes_list)-1):
                    if(yes_list[j]+no_list[j]<yes_list[j+1]+no_list[j+1]):
                        yes_list[j], yes_list[j+1] = yes_list[j+1], yes_list[j]
                        no_list[j], no_list[j+1] = no_list[j+1], no_list[j]
                        keys[j], keys[j+1] = keys[j+1], keys[j]
    
    return keys,no_list,yes_list
    
def getStatistic(ds,attribute):
    column=ds[attribute]
    statistics={}
    statistics["Min"]=min(column)
    statistics["1st Q."]=np.quantile(column, 0.25)
    statistics["Median"]=np.quantile(column, 0.5)
    statistics["Mean"]=(sum(column)/len(column))
    statistics["Std"]=np.std(column, axis=0)
    statistics["3st Q."]=np.quantile(column, 0.75)
    statistics["Max"]=max(column)
    return statistics  
    
def drawStatisticsTable(data,columns,rows):
    data_tmp=np.array(data).T.tolist()
    for i in range(0,len(rows)):
        for j in range(0,len(columns)):
            if(int(data_tmp[i][j])==data_tmp[i][j]):
                data_tmp[i][j]=int(data_tmp[i][j])
            else:
                data_tmp[i][j]=round(data_tmp[i][j],3)
    Rcolors = plt.cm.BuPu(np.linspace(0.3, 0.3, len(rows)))
    Ccolors = plt.cm.BuPu(np.linspace(0.3, 0.3, len(columns)))
    cell_text = []
    for row in data_tmp:
        cell_text.append(row)
    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=Rcolors,
                      colColours=Ccolors,
                      colLabels=columns,
                      cellLoc="center",
                      loc='center')
    the_table.scale(0.6, 3)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
