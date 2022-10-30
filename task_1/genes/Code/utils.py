import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def read_data_genes(datapath = "./data/Genes/data.csv", labelpath = "./data/Genes/labels.csv"):
    data = pd.read_csv(datapath)
    labels = pd.read_csv(labelpath)
    return data, labels

def data_split_genes(data, labels):
    X = data.iloc[: , 1:]
    y = labels['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 2020, stratify=y)
    
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    y_train = np.array(y_train)
    return X_train_scaled, X_test_scaled, y_train, y_test