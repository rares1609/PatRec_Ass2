from Code.utils import *
from Code.features import *

def classification_random_forest(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(max_depth=2, max_leaf_nodes=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy' + str(accuracy_score(y_test, y_pred)))
    
    