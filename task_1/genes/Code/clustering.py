from Code.utils import *
from Code.features import *
from Code.classification import *

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score
from pandas import concat,crosstab,DataFrame


def cluster_label(cluster_label):
    if cluster_label == 0:
        return 'BRCA'
    if cluster_label == 1:
        return 'PRAD'
    if cluster_label == 2:
        return 'LUAD'
    if cluster_label == 3:
        return 'COAD'
    if cluster_label == 4:
        return 'KIRC'

def cluster_label_balanced(cluster_label):
    if cluster_label == 0:
        return 'COAD'
    if cluster_label == 1:
        return 'BRCA'
    if cluster_label == 2:
        return 'KIRC'
    if cluster_label == 3:
        return 'PRAD'
    if cluster_label == 4:
        return 'LUAD'

def cluster_label_lda(cluster_label):
    if cluster_label == 0:
        return 'COAD'
    if cluster_label == 1:
        return 'LUAD'
    if cluster_label == 2:
        return 'PRAD'
    if cluster_label == 3:
        return 'KIRC'
    if cluster_label == 4:
        return 'BRCA'

def KMeans_clustering(data, labels, scenario = 'original'):
    model = KMeans(n_clusters = 5, random_state=12)
    model.fit(data)
    KM_labels = DataFrame()
    KM_labels['1'] = model.labels_
    if scenario == 'original':
        predictions = KM_labels['1'].apply(cluster_label)
    elif scenario == 'balanced':
        predictions = KM_labels['1'].apply(cluster_label_balanced)
    elif scenario == 'reduced':
        predictions = KM_labels['1'].apply(cluster_label_lda)
    
    cm = confusion_matrix(labels, predictions)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(labels, predictions)))
    print('F1 score: ' + str(f1_score(labels, predictions, average='weighted')))