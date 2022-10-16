import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from Code.analysis import *
from sklearn.decomposition import PCA
from itertools import product



data, labels = read_data_genes()





'''
def generate_features_genes():
    features = []
    for i in range(0, 20531):
        features.append('gene_'+str(i))
    return features

def standardize_data_genes():
    data, labels = read_data_genes()
    features = generate_features_genes()
    data2 = data.assign(Class = labels['Class'])
    x = data2.loc[:, features].values
    y = data2.loc[:,['Class']].values
    x = StandardScaler().fit_transform(x)
    return x, y

def PCA_Projection_genes():
    data, labels = read_data_genes()
    data2 = data.assign(Class = labels['Class'])
    x, y = standardize_data_genes()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, data2[['Class']]], axis = 1)
    return finalDf

def plot_projection_genes():
    df = PCA_Projection_genes()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['PRAD', 'LUAD', 'BRCA', 'KIRC', 'COAD']
    colors = ['r', 'g', 'b', 'y', 'm']
    for target, color in zip(targets,colors):
        indicesToKeep = df['Class'] == target
        ax.scatter(df.loc[indicesToKeep, 'principal component 1']
               , df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()



'''

def divide_into_features_labels_genes():
    #data, labels = read_data_genes()
    Y = labels['Class']
    data2 = data.iloc[: , 1:]
    Y = LabelEncoder().fit_transform(Y)
    X = StandardScaler().fit_transform(data2)
    return X, Y


def forest_test():  # Random Forest test for raw dataset
    X, Y = divide_into_features_labels_genes()
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, 
                                                        test_size = 0.30, 
                                                        random_state = 101)
    print("Random Forest for raw dataset")
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)
    print(time.process_time() - start)
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))

def principal_component_analysis():
    X, Y = divide_into_features_labels_genes()
    data2 = data.assign(Class = labels['Class'])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
    PCA_df = pd.concat([PCA_df, data2['Class']], axis = 1)
    #PCA_df['Class'] = LabelEncoder().fit_transform(PCA_df['Class'])
    return PCA_df, X_pca, Y

def plot_2D_PCA():
    PCA_df, _, _ = principal_component_analysis()
    figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

    classes = [0, 1, 2, 3, 4]
    targets = ['PRAD', 'LUAD', 'BRCA', 'KIRC', 'COAD']
    colors = ['r', 'g', 'b', 'y', 'm']
    for clas, color in zip(targets, colors):
        plt.scatter(PCA_df.loc[PCA_df['Class'] == clas, 'PC1'], 
                    PCA_df.loc[PCA_df['Class'] == clas, 'PC2'], 
                    c = color)
    
    plt.xlabel('Principal Component 1', fontsize = 12)
    plt.ylabel('Principal Component 2', fontsize = 12)
    plt.title('2D PCA', fontsize = 15)
    plt.legend(['PRAD', 'LUAD', 'BRCA', 'KIRC', 'COAD'])
    plt.grid()
    plt.show()








