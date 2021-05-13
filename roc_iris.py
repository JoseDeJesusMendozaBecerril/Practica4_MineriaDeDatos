
import numpy as np 
import os 

import matplotlib as mpl 
import matplotlib.pyplot as plt


#IRIS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

#WINE
from sklearn.datasets import load_wine

#CANCER
from sklearn.datasets import load_breast_cancer

#Arbol de decision
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor


from sklearn.model_selection import train_test_split #separa data

from sklearn import *

#PARTICIONES
from matplotlib.colors import ListedColormap

#MATRIZ CONFUSION
from sklearn.metrics import confusion_matrix

#ESPACIO ROC
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

#Iterable
from itertools import cycle

#Regresion Lineal
from sklearn.linear_model import LogisticRegression

#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#KNN
from sklearn.neighbors import KNeighborsClassifier

#ESCALAR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier



def main():
    ##-----------------------------------------DATA SET IRIS --------------------------------
    #CARGO DATA SET
    X,y = datasets.load_iris(return_X_y=True)
    

    #PARTIMOS DATA
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    tree_clf = tree.DecisionTreeClassifier()

    #PRUEBA MODELO - TRAIN
    tree_clf = tree_clf.fit(X_train,y_train)

    #ESPACIO ROC
    
    
    # Binarize the output
    y = label_binarize(y, classes=[0,1,2])
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

    # Arboles
    classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    print(y_score)

    #KNN
    classifier2 = OneVsRestClassifier(KNeighborsClassifier(algorithm='brute',n_neighbors=10))
    y_score2 = classifier2.fit(X_train, y_train).predict_proba(X_test)

    #Regresion logistica
    classifier3 = OneVsRestClassifier(LogisticRegression())
    y_score3 = classifier3.fit(X_train, y_train).predict_proba(X_test)

    #Naive bayes
    classifier4 = OneVsRestClassifier(GaussianNB())
    y_score4 = classifier4.fit(X_train, y_train).predict_proba(X_test)



    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    

    fpr2 = dict()
    tpr2= dict()

    fpr3 = dict()
    tpr3 = dict()

    fpr4 = dict()
    tpr4 = dict()

    roc_auc = dict()
    roc_auc2 = dict()
    roc_auc3 = dict()
    roc_auc4 = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        fpr2[i], tpr2[i], _ = roc_curve(y_test[:, i], y_score2[:, i])
        fpr3[i], tpr3[i], _ = roc_curve(y_test[:, i], y_score3[:, i])
        fpr4[i], tpr4[i], _ = roc_curve(y_test[:, i], y_score4[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        roc_auc2[i] = auc(fpr2[i], tpr2[i])
        roc_auc3[i] = auc(fpr3[i], tpr3[i])
        roc_auc4[i] = auc(fpr4[i], tpr4[i])

    
    colors = ['red','blue','darkorange','green']
    names = ['Arboles','KNN','regresion','Naive Bayes']
    plt.figure()
    lw = 1

    #COMPARO EFECTIVIDAD CON UNA SOLA CLASE 
    plt.plot(fpr[2], tpr[2], color=colors[0], lw=1, label= names[0])
    plt.plot(fpr2[2], tpr2[2], color=colors[1], lw=1, label= names[1])
    plt.plot(fpr3[2], tpr3[2], color=colors[2], lw=1, label= names[2])
    plt.plot(fpr4[2], tpr4[2], color=colors[3], lw=1, label= names[3])
        
        


    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


    






main()







