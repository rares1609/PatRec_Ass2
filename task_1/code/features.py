from Code.analysis import extract_classses_genes
from Code.utils import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from IPython.display import display
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score

def resample_data(data, labels):
    data['Class'] = labels['Class']
    del labels
    val = data['Class'].value_counts().values.tolist()
    BRCA = data[data.Class=='BRCA']
    KIRC = resample(data[data.Class=='KIRC'], 
                            replace=True,
                                n_samples=val[0],
                                    random_state=12)
    LUAD = resample(data[data.Class=='LUAD'], 
                            replace=True,
                                n_samples=val[0],
                                    random_state=12)
    PRAD = resample(data[data.Class=='PRAD'], 
                            replace=True,
                                n_samples=val[0],
                                    random_state=12)
    COAD = resample(data[data.Class=='COAD'], 
                            replace=True,
                                n_samples=val[0],
                                    random_state=12)
    data = pd.concat([BRCA, KIRC, LUAD, PRAD, COAD])
    print("Values resampled to:")
    print(data['Class'].value_counts())
    labels = pd.DataFrame()
    labels['Class'] = data['Class']
    data.drop('Class', inplace=True, axis=1)

    return data,labels

data, labels = read_data_genes()
#data, labels = resample_data(data,labels)
data2 = data.iloc[: , 1:]




    


def data_split_genes():
    data, labels = read_data_genes()
    data, labels = resample_data(data,labels)
    X = data.iloc[: , 1:]
    y = labels['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 2020, stratify=y)
    return X_train, X_test, y_train, y_test

def scale_data_genes():
    X_train, X_test, y_train, y_test = data_split_genes()
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    y_train = np.array(y_train)
    return X_train_scaled, X_test_scaled, y_train, y_test

def baseline_random_forest():
    X_train_scaled, X_test_scaled, y_train, y_test = scale_data_genes()
    rfc = RandomForestClassifier(max_depth= 5, n_jobs= -1)
    rfc.fit(X_train_scaled, y_train)
    #display(rfc.score(X_train_scaled, y_train))
    predictions = rfc.predict(X_test_scaled)
    display(balanced_accuracy_score(y_test, predictions))
    feats = {}
    for feature, importance in zip(data2.columns, rfc.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    #sns.set(font_scale = 5)
    #sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
    #fig, ax = plt.subplots()
    #fig.set_size_inches(30,15)
    #sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    #plt.xlabel('Importance', fontsize=25, weight = 'bold')
    #plt.ylabel('Features', fontsize=25, weight = 'bold')
    #plt.title('Feature Importance', fontsize=25, weight = 'bold')
    #display(plt.show())
    display(importances)

def PCA_test():
    pca_test = PCA(n_components=30)
    X_train_scaled, X_test_scaled, y_train, y_test = scale_data_genes()
    pca_test.fit(X_train_scaled)
    sns.set(style='whitegrid')
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.title("PCA test for 'Genes' dataset")
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
    display(plt.show())

    evr = pca_test.explained_variance_ratio_
    cvr = np.cumsum(pca_test.explained_variance_ratio_)
    pca_df = pd.DataFrame()
    pca_df['Cumulative Variance Ratio'] = cvr
    pca_df['Explained Variance Ratio'] = evr
    display(pca_df.head(10))
    
def PCA_10_components():
    X_train_scaled, X_test_scaled, y_train, y_test = scale_data_genes()
    pca = PCA(n_components=10)
    pca.fit(X_train_scaled)
    X_train_scaled_pca = pca.transform(X_train_scaled)
    X_test_scaled_pca = pca.transform(X_test_scaled)
    pca_df = pd.DataFrame(data = X_train_scaled_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
    pca_df = pd.concat([pca_df, labels['Class']], axis = 1)
    return pca_df, X_train_scaled_pca, y_train






