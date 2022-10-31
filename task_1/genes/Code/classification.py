from Code.utils import *
from Code.features import *

import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score


def classification_random_forest(X_train, y_train, X_test, y_test, param = None):
    if param:
        classifier = RandomForestClassifier(
            random_state=12, 
            n_estimators=param['n_estimators'], 
            max_depth=param['max_depth'], 
            max_features=param['max_features'], 
            criterion= param['criterion']
        )
    else:
        classifier = RandomForestClassifier(
            random_state=12, 
            n_estimators=param['n_estimators'], 
            max_depth=param['max_depth'], 
            max_features=param['max_features'], 
            criterion= param['criterion']
        )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    print('F1 score: ' + str(f1_score(y_test, y_pred, average='weighted')))
    #print('ROC-AUC score: ' + str(roc_auc_score(y_test, y_pred, average='weighted')))

#{'C': 1e-05, 'max_iter': 50, 'penalty': 'l2', 'solver': 'newton-cg'}
def classification_logistic_regression(X_train, y_train, X_test, y_test, param = None):
    if param:
        logreg = LogisticRegression(
            random_state=12,
            max_iter=param['max_iter'], 
            solver=param['solver'],
            penalty=param['penalty'], 
            C=param['C']
            )
    # in case the grid search is did not run, the values from a previous gridsearch are used
    else:
        logreg = LogisticRegression(
            random_state=12,
            max_iter=50, 
            solver='newton-cg',
            penalty='l2', 
            C=1e-05
            )
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    print('F1 score: ' + str(f1_score(y_test, y_pred, average='weighted')))
    #print('ROC-AUC score: ' + str(roc_auc_score(y_test, y_pred, average='weighted')))

def classification_knn(X_train, y_train, X_test, y_test, param= None):
    if param:
        knn = KNeighborsClassifier(
            n_neighbors=param['n_neighbors'],
            metric=param['metric'], 
            weights=param['weights']
            )
    else:
        knn = KNeighborsClassifier(
            n_neighbors=param['n_neighbors'],
            metric=param['metric'], 
            weights=param['weights']
            )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
    print('F1 score: ' + str(f1_score(y_test, y_pred, average='weighted')))
    #print('ROC-AUC score: ' + str(roc_auc_score(y_test, y_pred, average='weighted')))