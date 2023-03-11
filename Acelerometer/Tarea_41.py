"""
Archivo Python resumido del Notebook Tarea_41 del repositorio
"""
#!/usr/bin/env python
# coding: utf-8

# # Tarea 4
# 
# # Emilio Hernández Arellano

# # Datos Originales

# ## Entrenamiento

# ## Preelimniares

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tsfel
import os


# In[2]:


list_class = {
    "MW":"Caminar",
    "MR":"Correr",
    "MJ":"Saltar",
    "WD": "Bajar escaleras",
    "WU": "Saltar escaleras",
    "FF" : "Caida hacia delante",
    "FS" : "Caida lateral",
    "FB" : "Caida hacia atras",
    "LD" : "Tumbado",
    "OT" : "Otras clases"
}


# In[6]:


from glob import glob


# In[7]:


#Obteniendo la clase
def label_class(file):
    basename = os.path.basename(file)
    label = basename.split("_")[-1].split(".")[0]
    if label in list_class:
        label_1 = list_class[label]
    else:
        print(file)
        label_1 = None
    return label_1


# In[8]:


#Filtro media movil opcional
def median_filter(column,n=10):
    c = pd.Series (column) .rolling (window = n) .mean (). iloc [n-1:]. values
    return c


# In[9]:


def feature_csv(df,cfg=tsfel.get_features_by_domain(),fs=200,n=None,window=False):
    x = df["x"]
    y = df["y"]
    z = df["z"]
    if window and n>=1:
        x = median_filter(x,n=n)
        y = median_filter(y,n=n)
        z = median_filter(z,n=n)
    xf = tsfel.time_series_features_extractor(cfg,x, fs)
    yf = tsfel.time_series_features_extractor(cfg,y, fs)
    zf = tsfel.time_series_features_extractor(cfg,z, fs)
    f = pd.concat([xf,yf,zf],axis=1)
    return f


# In[94]:


def load_data(path,train=True,n=None,window=False,fs=200):
    features = []
    if train:
        labels = []
        for file in glob(path):
            print(file)
            label = label_class(file)
            file_read = pd.read_csv(file,names=["x","y","z","time"])
            feature = feature_csv(file_read,n=n,window=window,fs=fs)
            feature = feature.values
            _,width = np.shape(feature)
            if label is not None and feature is not None and width == 1167:
                labels.append(label)
                features.extend(feature)
            elif label is None:
                print(file)
        labels = np.array(labels)
        features = np.array(features)
        return features,labels
    else:
        for file in glob(path):
            file_read = pd.read_csv(file)
            """
            Dado que los 35 archivos aparecían ya con 
            las cabeceras, X,Y,Z, tiempo, se omite el paso
            En caso contrario de no tener cabecera:
            pd.read_csv(file,names=["x","y","z","time"])
            """
            feature = feature_csv(file_read,n=n,window=window)
            feature = feature.values
            _,width = np.shape(feature)
            if feature is not None and width == 1167:
                features.extend(feature)
        return np.array(features)


# In[11]:


path_train = "C:/Users/millo/Documents/CICESE/Ciencia de Datos/Ejercicio_4/Train/Original/*/*.csv"
X,y = load_data(path_train)

# ## F1 SCORE

# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


# ### Random Forest

# In[14]:


#Señal sin media movil Random Forest
forest_int_wmean = RandomForestClassifier(n_estimators=100)
Kfold = KFold(n_splits=10,random_state=42,shuffle=True)
y_test_indx,y_pred = [],[]
scores = []

for i,(train_indx,test_indx) in enumerate (Kfold.split(X,y)):
    y_test_indx.append(test_indx)
    X_train,X_test = X[train_indx],X[test_indx]
    y_train,y_test = y[train_indx], y[test_indx]
    forest_int_wmean.fit(X_train,y_train) #Entrenando random forest
    ypred = forest_int_wmean.predict(X_test)
    score = forest_int_wmean.score(X_test,y_test)
    scores.append(score)
    y_pred.append(ypred)

print("Con el modelo de Random Forest se tiene un accuracy de %0.2f y una std: %0.2f" 
% (np.mean(scores), np.std(scores) * 2))
y_pred1 = np.concatenate(y_pred).tolist()
y_index1 = np.concatenate(y_test_indx).tolist()
tuples = list(zip(y_index1,y_pred1))
sorted_tuples = sorted(tuples)
y_index1,y_pred1 = zip(*sorted_tuples)
print(classification_report(y,y_pred1))


# ### Desicion Tree

# In[15]:


# Desicion Tree con señal original sin media movil

tree_int_wmean = tree.DecisionTreeClassifier()
y_test_indx,y_pred = [],[]
scores = []

for i,(train_indx,test_indx) in enumerate (Kfold.split(X,y)):
    y_test_indx.append(test_indx)
    X_train,X_test = X[train_indx],X[test_indx]
    y_train,y_test = y[train_indx], y[test_indx]
    tree_int_wmean.fit(X_train,y_train) 
    ypred = tree_int_wmean.predict(X_test)
    #print(X_train.shape)
    score = tree_int_wmean.score(X_test,y_test)
    scores.append(score)
    y_pred.append(ypred)

print("Con el modelo de Desicion Tree se tiene un accuracy de %0.2f y una std: %0.2f" 
% (np.mean(scores), np.std(scores) * 2))
y_pred1 = np.concatenate(y_pred).tolist()
y_index1 = np.concatenate(y_test_indx).tolist()
tuples = list(zip(y_index1,y_pred1))
sorted_tuples = sorted(tuples)
y_index1,y_pred1 = zip(*sorted_tuples)
print(classification_report(y,y_pred1))

# In[16]:


# Obteniendo las clases de los archivos muestras

path_eval = "C:/Users/millo/Documents/CICESE/Ciencia de Datos/Ejercicio_4/test/*.csv"
Y = load_data(path_eval,train=False)


# In[17]:


YPRED = forest_int_wmean.predict(Y)
name_forest = "ForestLabels.txt"
path_txt = "C:/Users/millo/Documents/CICESE/Ciencia de Datos/Ejercicio_4/"+ name_forest
np.savetxt(path_txt,YPRED,fmt="%s")


# In[18]:


YPRED_TREE = tree_int_wmean.predict(Y)
name_forest = "TreeLabels.txt"
path_txt = "C:/Users/millo/Documents/CICESE/Ciencia de Datos/Ejercicio_4/"+ name_forest
np.savetxt(path_txt,YPRED_TREE,fmt="%s")


