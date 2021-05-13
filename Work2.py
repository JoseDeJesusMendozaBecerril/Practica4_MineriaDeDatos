
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


mpl.rc('axes', labelsize=4)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID )
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id,tight_layout=True,fig_extension="png", resolution=300):
    path= os.path.join(IMAGES_PATH,fig_id + "." + fig_extension)
    print("Saving figure", fig_id)

    if tight_layout:
        plt.tight_layout()
    
    plt.savefig(path, format=fig_extension ,dpi=resolution)

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 1, 3], iris=True, legend=False,
plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)



def main():


    ##-----------------------------------------DATA SET CANCER --------------------------------
    data_cancer = load_breast_cancer()
    #print(data_cancer)

    #DEFINO ATRIBUTOS Y VALORES DE SALIDA
    X = data_cancer.data  # we only take the first two features.
    y = data_cancer.target

    #PARTIMOS DATA
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    #GENERACION DEL MODELO
    tree_clf = tree.DecisionTreeClassifier(max_depth=5)
    
    #PRUEBA MODELO - TRAIN
    tree_clf = tree_clf.fit(X_train,y_train)

    #PRUEBA MODELO - TEST
    tree_clf2 = tree_clf.fit(X_test,y_test)


    #CALCULO DE LA PRECISION DEL MODELO 

    print("DATA SET CANCER") 


    precision = tree_clf.score(X_train,y_train)
    print("Precision con datos entrenamiento: " , precision)

    precision2 = tree_clf2.score(X_test,y_test)
    print("Precision con datos prueba: " , precision2)


    #ESPACIO ROC
    y_score1 = tree_clf.predict_proba(X_test)[:,1]
    false_positive_rate1 , true_positive_rate1,threshold1 = roc_curve(y_test,y_score1)
    print("roc_auc_score data set cancer" , roc_auc_score(y_test,y_score1))


    knn = KNeighborsClassifier(algorithm='brute',n_neighbors=75)
    knn.fit(X_train,y_train) 

    y_score2 = knn.predict_proba(X_test)[:,1]

    false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)
    print('roc_auc_score for KNN: ', roc_auc_score(y_test, y_score2))


    NB = GaussianNB()
    NB.fit(X_train, y_train)
    y_score3 = NB.predict_proba(X_test)[:,1]

    false_positive_rate3, true_positive_rate3, threshold3 = roc_curve(y_test, y_score3)
    print('roc_auc_score for Naive Bayes: ', roc_auc_score(y_test, y_score3))


    plt.subplots(1, figsize=(10,10))
    plt.title('ROC for data cancer')
    
    plt.plot(false_positive_rate1, true_positive_rate1,color="blue")
    plt.plot(false_positive_rate2, true_positive_rate2, color="red")
    plt.plot(false_positive_rate3, true_positive_rate3, color="green")

    plt.plot([0, 1], ls="solid")
    """ plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7") """
    
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(["DecisionTreeClassifier", "KNN"," Naive Bayes"])
    plt.show()

main()







