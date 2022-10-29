from Code.utils import *
from Code.features import *
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

'''

def logistic_regression_genes():
    X_pca = PCA_Projection_genes()
    print(X_pca)

    # Make train and test sets

    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, X_pca['Class'], test_size=0.20, 
                                                            shuffle=True, random_state=2)

    # Initialize the logistic regression model

    clf = LogisticRegression(max_iter=2500)

    # Train the model

    clf.fit(X_train_pca, y_train)

    # Make predictions

    y_pred = clf.predict(X_test_pca) # Predictions
    y_true = y_test # True values

    # Measure accuracy

    print("Train accuracy:", np.round(accuracy_score(y_train, 
                                                 clf.predict(X_train_pca)), 2))
    print("Test accuracy:", np.round(accuracy_score(y_true, y_pred), 2))

    # Make the confusion matrix

    cf_matrix = confusion_matrix(y_true, y_pred)
    print("\nTest confusion_matrix")
    sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)

'''

def random_forest():
    data, labels = read_data_genes()
    data2 = data.iloc[: , 1:]
    PCA_df, X_pca, Y = principal_component_analysis()
    X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y, 
                                                                        test_size = 0.30, 
                                                                        random_state = 101)
#    X, Y = divide_into_features_labels_genes()
#    X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X, Y, 
#                                                                        test_size = 0.30, 
#                                                                        random_state = 101)
    print("Random Forest for PCA dataset")
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Reduced,Y_Reduced)

    print(time.process_time() - start)
    predictionforest = trainedforest.predict(X_Test_Reduced)
    print(confusion_matrix(Y_Test_Reduced,predictionforest))
    print(classification_report(Y_Test_Reduced,predictionforest))

    x_min, x_max = X_Reduced[:, 0].min() - 1, X_Reduced[:, 0].max() + 1
    y_min, y_max = X_Reduced[:, 1].min() - 1, X_Reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = trainedforest.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)
    plt.scatter(X_Reduced[:, 0], X_Reduced[:, 1], c=Y_Reduced, s=20, edgecolor='k')
    plt.xlabel('Principal Component 1', fontsize = 12)
    plt.ylabel('Principal Component 2', fontsize = 12)
    plt.title('Random Forest', fontsize = 15)
    plt.show()


