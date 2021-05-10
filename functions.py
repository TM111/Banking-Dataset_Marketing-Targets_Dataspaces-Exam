import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

attributes_index = {
  "age": 0,
  "job": 1,
  "marital": 2,
  "education": 3,
  "default": 4,
  "balance": 5,
  "housing": 6,
  "loan": 7,
  "contact": 8,
  "day": 9,
  "month": 10,
  "duration": 11,
  "campaign": 12,
  "pdays": 13,
  "previous": 14,
  "poutcome": 15,
  "Class": 16
}

catToNumDict = {
   "no": 0,
  "yes": 1,
  
  "telephone": 0,
  "cellular": 1,
  
  "primary": 0,
  "secondary": 1,
  "tertiary": 3,
  
}

def categoricalToNumeric(ds):
    dataset=ds
    for i in range(0,len(dataset)):
        for j in range(0,len(dataset[0])):
            if(str(dataset[i][j]) in catToNumDict.keys()):
                dataset[i][j]=catToNumDict[str(dataset[i][j])]
    return dataset
    
def getAttributes():
    return list(attributes_index.keys())
def formatDataset(csv,test,l,r):
    ds=[]
    lenght=len(csv)
    if(test==1):
        if(r==1):
            csv=csv.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)
        lenght=l
    for i in range(0,lenght):
        tmp_list=[]
        string=""
        for c in csv.iloc[i][0]:
            if(c!='"'):
                string=string+c
        for e in string.split(";"):
            try:
                tmp_list.append(int(e))
            except:
                tmp_list.append(e)
        ds.append(tmp_list)
    return ds

def removeAttribute(ds,attribute):
    index=attributes_index[attribute]
    for instance in ds:
        instance.pop(index)


def removeAttributeValue(ds,attribute,value,test):
    if(test==1):
        return
    if(type(attribute)==type(2)):
        index=attribute
    else:
        index=attributes_index[attribute]
    tmp_ds=[]
    for i in range(0,len(ds)):
        if(str(ds[i][index])!=str(value)):
            tmp_ds.append(ds[i])
    ds[:]=tmp_ds
        
def getElement(ds, index,attribute):
    return ds[index][attributes_index[attribute]]

def getColumn(ds,attribute):
    index=attributes_index[attribute]
    column=[]
    for instance in ds:
        column.append(instance[index])
    return column

def getColumnClass(ds,attribute,y):
    index=attributes_index[attribute]
    class_index=attributes_index["Class"]
    column=[]
    for instance in ds:
        if(instance[class_index]==y):
            column.append(instance[index])
    return column

def getOccurrences(ds,attribute,normalize=0,order=0):
    class_index=attributes_index["Class"]
    column=getColumn(ds,attribute)
    total_sum=len(column)
    values={}
    for v in column:
        if(v not in values):
            values[v]=[0,0]
            
    for instance in ds:
        value=instance[attributes_index[attribute]]
        if(instance[class_index]=="no" or instance[class_index]==0):
            values[value]=[values[value][0]+1,values[value][1]]
        else:
            values[value]=[values[value][0],values[value][1]+1]
    keys=list(values.keys())
    no_list=[]
    yes_list=[]
    for k in values.keys():
        sum=values[k][0]+values[k][1]
        if(normalize==0):
            sum=1
        no_list.append(values[k][0]/sum)
        yes_list.append(values[k][1]/sum)
    
    if(order==1):
        for i in range(0,len(yes_list)-1):
            for j in range(0,len(yes_list)-1):
                if(yes_list[j]>yes_list[j+1]):
                    yes_list[j], yes_list[j+1] = yes_list[j+1], yes_list[j]
                    no_list[j], no_list[j+1] = no_list[j+1], no_list[j]
                    keys[j], keys[j+1] = keys[j+1], keys[j]
    return keys,no_list,yes_list

    
def getStatistic(ds,attribute):
    column=getColumn(ds,attribute)
    statistics={}
    statistics["Min"]=min(column)
    statistics["1st Q."]=np.quantile(column, 0.25)
    statistics["Median"]=np.quantile(column, 0.5)
    statistics["Mean"]=(sum(column)/len(column))
    statistics["Std"]=np.std(column, axis=0)
    statistics["3st Q."]=np.quantile(column, 0.75)
    statistics["Max"]=max(column)
    return statistics  
    
def drawTable(data,columns,rows):
    data=np.array(data).T.tolist()
    for i in range(0,len(rows)):
        for j in range(0,len(columns)):
            if(int(data[i][j])==data[i][j]):
                data[i][j]=int(data[i][j])
            else:
                data[i][j]=round(data[i][j],3)
    Rcolors = plt.cm.BuPu(np.linspace(0.3, 0.3, len(rows)))
    Ccolors = plt.cm.BuPu(np.linspace(0.3, 0.3, len(rows)))
    cell_text = []
    for row in data:
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
