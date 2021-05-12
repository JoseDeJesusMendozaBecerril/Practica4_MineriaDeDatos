
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
    ##-----------------------------------------DATA SET IRIS --------------------------------
    #CARGO DATA SET
    iris = datasets.load_iris()
    #print("EL IRIS",iris)
    #DEFINO ATRIBUTOS Y VALORES DE SALIDA
    X = iris.data[:,:2]  # we only take the first two features.
    y = iris.target

    

    #PARTIMOS DATA
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    #GENERACION DEL MODELO
    tree_clf = tree.DecisionTreeClassifier()
    
    #PRUEBA MODELO - TRAIN
    tree_clf = tree_clf.fit(X_train,y_train)

    
    export_graphviz(
            tree_clf,
            out_file = os.path.join(IMAGES_PATH, "iris_tree_train.dot"),
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
    )
    Source.from_file(os.path.join(IMAGES_PATH,"iris_tree_train.dot"))

    #PRUEBA MODELO - TEST
    tree_clf2 = tree_clf.fit(X_test,y_test)

    #DIBUJAR EL MODELO DEL ARBOL    
    export_graphviz(
            tree_clf2,
            out_file = os.path.join(IMAGES_PATH, "iris_tree_test.dot"),
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
    )
    Source.from_file(os.path.join(IMAGES_PATH,"iris_tree_test.dot"))

    

    #CALCULO DE LA PRECISION DEL MODELO
    print("DATA SET IRIS") 
    precision = tree_clf.score(X_train,y_train)
    print("Precision con datos entrenamiento: " , precision)

    precision2 = tree_clf2.score(X_test,y_test)
    print("Precision con datos prueba: " , precision2)
    
    #VISUALIZAR LAS PARTICIONES GENERADAS POR CADA SPLIT (SOLO PARA DATA SET IRIS)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    plt.figure(figsize=(8,4))
    plot_decision_boundary(tree_clf,X,y)
    plt.plot([2.45,2.45],[0,3],"k-",linewidth=2)
    plt.plot([2.45,7.5],[1.75,1.75],"k--",linewidth=2)
    plt.text(1.40,1.0,"Depth=0",fontsize=15)
    plt.text(3.2,1.80,"Depth=1",fontsize=13)
    save_fig("decision_tree_decision_boundaries_plot_completo")
    plt.show()


    plot_decision_boundary(tree_clf,X_train,y_train)
    plt.plot([2.45,2.45],[0,3],"k-",linewidth=2)
    plt.plot([2.45,7.5],[1.75,1.75],"k--",linewidth=2)
    plt.text(1.40,1.0,"Depth=0",fontsize=15)
    plt.text(3.2,1.80,"Depth=1",fontsize=13)
    save_fig("decision_tree_decision_boundaries_plot_train")
    plt.show()

    plot_decision_boundary(tree_clf,X_test,y_test)
    plt.plot([2.45,2.45],[0,3],"k-",linewidth=2)
    plt.plot([2.45,7.5],[1.75,1.75],"k--",linewidth=2)
    plt.text(1.40,1.0,"Depth=0",fontsize=15)
    plt.text(3.2,1.80,"Depth=1",fontsize=13)
    save_fig("decision_tree_decision_boundaries_plot_test")
    plt.show()



    #MATRIZ DE CONFUSION
    print("Matriz de confusion")
    y_pred = tree_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test,y_pred)
    print(conf_matrix)


    


    #DESEMPEÑO RESPECTO A Regresion logistica - KNN - Naive Bayes 
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train,y_train)


    clf = LogisticRegression()
    knn = KNeighborsClassifier(algorithm='brute',n_neighbors=20)
    nb = GaussianNB()

    
    knn.fit(X_train,y_train)
    nb.fit(X_train,y_train)

    score_clf = pipe.score(X_train,y_train)
    print("Score clf",score_clf)
    
    score_knn = knn.score(X_train,y_train)
    print("Score knn" , score_knn)


    score_nb = nb.score(X_train,y_train)
    print("Score nb" , score_nb)


     #ESPACIO ROC
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

    classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    lw=2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()







    ##-----------------------------------------DATA SET WINE --------------------------------
    data_wine = load_wine()
    #print(data_wine)

    #DEFINO ATRIBUTOS Y VALORES DE SALIDA
    X = data_wine.data  # we only take the first two features.
    y = data_wine.target

    #PARTIMOS DATA
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    #GENERACION DEL MODELO
    tree_clf = tree.DecisionTreeClassifier()
    
    #PRUEBA MODELO - TRAIN
    tree_clf = tree_clf.fit(X_train,y_train)

    #DIBUJAR EL MODELO DEL ARBOL
    export_graphviz(
            tree_clf,
            out_file = os.path.join(IMAGES_PATH, "wine_tree_train.dot"),
            feature_names=data_wine.feature_names,
            class_names=data_wine.target_names,
            rounded=True,
            filled=True
    )
    Source.from_file(os.path.join(IMAGES_PATH,"wine_tree_train.dot"))


    #PRUEBA MODELO - TEST
    tree_clf2 = tree_clf.fit(X_test,y_test)
    
    export_graphviz(
            tree_clf2,
            out_file = os.path.join(IMAGES_PATH, "wine_tree_test.dot"),
            feature_names=data_wine.feature_names,
            class_names=data_wine.target_names,
            rounded=True,
            filled=True
    )
    Source.from_file(os.path.join(IMAGES_PATH,"wine_tree_test.dot"))

    
    #CALCULO DE LA PRECISION DEL MODELO 

    print("DATA SET WINE") 

    precision = tree_clf.score(X_train,y_train)
    print("Precision con datos entrenamiento: " , precision)

    precision2 = tree_clf2.score(X_test,y_test)
    print("Precision con datos prueba: " , precision2)

    #MATRIZ DE CONFUSION
    print("Matriz de confusion")
    y_pred = tree_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test,y_pred)
    print(conf_matrix)


    
    #DESEMPEÑO RESPECTO A Regresion logistica - KNN - Naive Bayes 
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train,y_train)


    clf = LogisticRegression()
    knn = KNeighborsClassifier(algorithm='brute',n_neighbors=20)
    nb = GaussianNB()

    
    knn.fit(X_train,y_train)
    nb.fit(X_train,y_train)

    score_clf = pipe.score(X_train,y_train)
    print("Score clf",score_clf)
    
    score_knn = knn.score(X_train,y_train)
    print("Score knn" , score_knn)


    score_nb = nb.score(X_train,y_train)
    print("Score nb" , score_nb)




     #ESPACIO ROC
    data_wine = datasets.load_wine()
    X = data_wine.data  
    y = data_wine.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

    classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    lw=2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()


    ##-----------------------------------------DATA SET CANCER --------------------------------
    data_cancer = load_breast_cancer()
    #print(data_cancer)

    #DEFINO ATRIBUTOS Y VALORES DE SALIDA
    X = data_cancer.data  # we only take the first two features.
    y = data_cancer.target

    #PARTIMOS DATA
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    #GENERACION DEL MODELO
    tree_clf = tree.DecisionTreeClassifier()
    
    #PRUEBA MODELO - TRAIN
    tree_clf = tree_clf.fit(X_train,y_train)

    export_graphviz(
            tree_clf,
            out_file = os.path.join(IMAGES_PATH, "cancer_tree_train.dot"),
            feature_names=data_cancer.feature_names,
            class_names=data_cancer.target_names,
            rounded=True,
            filled=True
    )
    Source.from_file(os.path.join(IMAGES_PATH,"cancer_tree_train.dot"))


    #PRUEBA MODELO - TEST
    tree_clf2 = tree_clf.fit(X_test,y_test)
    
    export_graphviz(
            tree_clf2,
            out_file = os.path.join(IMAGES_PATH, "cancer_tree_test.dot"),
            feature_names=data_cancer.feature_names,
            class_names=data_cancer.target_names,
            rounded=True,
            filled=True
    )
    Source.from_file(os.path.join(IMAGES_PATH,"cancer_tree_test.dot"))


    #CALCULO DE LA PRECISION DEL MODELO 

    print("DATA SET CANCER") 


    precision = tree_clf.score(X_train,y_train)
    print("Precision con datos entrenamiento: " , precision)

    precision2 = tree_clf2.score(X_test,y_test)
    print("Precision con datos prueba: " , precision2)

    #MATRIZ DE CONFUSION
    print("Matriz de confusion")
    y_pred = tree_clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test,y_pred)
    print(conf_matrix)

    #ESPACIO ROC
    y_score1 = tree_clf.predict_proba(X_test)[:,1]
    false_positive_rate1 , true_positive_rate1,threshold1 = roc_curve(y_test,y_score1)
    print("roc_auc_score data set cancer" , roc_auc_score(y_test,y_score1))

    plt.plot(1)
    plt.title("Data set cancer")
    plt.plot(false_positive_rate1,true_positive_rate1,color="red")

    plt.plot([0,1] , [0,1] , color="navy" , lw=lw , linestyle='--')

    plt.show()

    #DESEMPEÑO RESPECTO A Regresion logistica - KNN - Naive Bayes 
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train,y_train)


    clf = LogisticRegression()
    knn = KNeighborsClassifier(algorithm='brute',n_neighbors=20)
    nb = GaussianNB()

    
    knn.fit(X_train,y_train)
    nb.fit(X_train,y_train)

    score_clf = pipe.score(X_train,y_train)
    print("Score clf",score_clf)
    
    score_knn = knn.score(X_train,y_train)
    print("Score knn" , score_knn)


    score_nb = nb.score(X_train,y_train)
    print("Score nb" , score_nb)

main()







