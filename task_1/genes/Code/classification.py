from Code.utils import *
from Code.features import *

import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score


def classification_random_forest(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(max_depth=2, max_leaf_nodes=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    print('F1 score: ' + str(f1_score(y_test, y_pred, average='weighted')))
    #print('ROC-AUC score: ' + str(roc_auc_score(y_test, y_pred, average='weighted')))

def classification_logistic_regression(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(random_state=16, max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    print('F1 score: ' + str(f1_score(y_test, y_pred, average='weighted')))
    #print('ROC-AUC score: ' + str(roc_auc_score(y_test, y_pred, average='weighted')))

def classification_knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    print('F1 score: ' + str(f1_score(y_test, y_pred, average='weighted')))
    #print('ROC-AUC score: ' + str(roc_auc_score(y_test, y_pred, average='weighted')))