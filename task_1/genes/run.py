import argparse
import sys
from pandas import concat, Series
from numpy import concatenate
from Code.analysis import plot_classes_histogram_genes,plot_PCA_components,select_n_components_lda
from Code.features import *
from Code.classification import *
from Code.clustering import *
from Code.augment import *
from Code.grid_search import grid_search_KNN, grid_search_forest, grid_search_logreg


def get_arguments():
    parser = argparse.ArgumentParser(description='Pattern Recogniton Task1')

    parser.add_argument('-no-analysis', action=argparse.BooleanOptionalAction, help='Do not Perform Analysis for data')
    parser.add_argument('-no-features', action=argparse.BooleanOptionalAction, help='Do not Perform Feature Selection')
    parser.add_argument('-no-classify', action=argparse.BooleanOptionalAction, help='Do not Perform Classification')
    parser.add_argument('-no-cluster', action=argparse.BooleanOptionalAction, help='Do not Perform Clustering')
    parser.add_argument('-no-search', action=argparse.BooleanOptionalAction, help='Do not Perform Grid Search')
    parser.add_argument('-no-evaluate', action=argparse.BooleanOptionalAction, help='DO not Perform Evaluation')

    
    arguments = parser.parse_args()
    print(arguments)
    return arguments

def init_best_param():
    # Declaring parameters as best parameters from a previous grid search, in case run is made with '-no-search'
    best_param_forest_balanced = {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 200}
    best_param_forest_pca = {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'sqrt', 'n_estimators': 200}
    best_param_forest_lda = {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 200}

    best_param_log_balanced = {'C': 1e-05, 'max_iter': 50, 'penalty': 'l2', 'solver': 'newton-cg'}
    best_param_log_pca = {'C': 1e-05, 'max_iter': 50, 'penalty': 'l2', 'solver': 'newton-cg'}
    best_param_log_lda = {'C': 1e-05, 'max_iter': 50, 'penalty': 'l2', 'solver': 'newton-cg'}

    best_param_knn_balanced = {'metric': 'cosine', 'n_neighbors': 7, 'weights': 'distance'}
    best_param_knn_pca = {'metric': 'minkowski', 'n_neighbors': 3, 'weights': 'distance'}
    best_param_knn_lda = {'metric': 'cosine', 'n_neighbors': 5, 'weights': 'uniform'}

    return best_param_forest_balanced, best_param_forest_pca, best_param_forest_lda, best_param_log_balanced, best_param_log_pca, best_param_log_lda, best_param_knn_balanced, best_param_knn_pca, best_param_knn_lda


def run(no_analysis, no_features, no_classify, no_cluster, no_search, no_evaluate):
    best_param_forest_balanced, best_param_forest_pca, best_param_forest_lda, best_param_log_balanced, best_param_log_pca, best_param_log_lda, best_param_knn_balanced, best_param_knn_pca, best_param_knn_lda = init_best_param()

    data, labels = read_data_genes()
    data_balanced, labels_balanced = resample_data(data, labels)
    X_train, X_test, y_train, y_test = data_split_genes(data_balanced, labels_balanced)
    if not(no_analysis):
        print("add all code for running analysis")
        plot_classes_histogram_genes(labels)
        plot_classes_histogram_genes(labels_balanced)
        plot_PCA_components(X_train)

    if not(no_features):
        print("add all code for running feature extraction")
        X_train_PCA, X_test_PCA = PCA_10_components(X_train, X_test)
        print("\nNumber of components for LDA: ", select_n_components_lda(data_balanced.iloc[: , 1:], labels_balanced['Class'], 0.95))
        X_train_LDA, X_test_LDA = LDA_n_components(X_train, X_test, y_train, y_test, select_n_components_lda(data_balanced.iloc[: , 1:], labels_balanced['Class'], 0.95))
        print("\n")

    
    if not(no_search):
        print("add all code for running grid search")
        print("\nGrid Search for Random Forest balanced data:")
        #best_param_forest_balanced = grid_search_forest(X_train,y_train)
        print("\n\nGrid Search for Random Forest PCA data:")
        #best_param_forest_pca = grid_search_forest(X_train_PCA,y_train)
        print("\n\nGrid Search for Random Forest LDA data:")
        #best_param_forest_lda = grid_search_forest(X_train_LDA,y_train)

        print("\nGrid Search for Logistic Regregresion balanced data:")
        best_param_log_balanced = grid_search_logreg(X_train,y_train)
        print("\n\nGrid Search for Logistic Regregresion PCA data:")
        best_param_log_pca = grid_search_logreg(X_train_PCA,y_train)
        print("\n\nGrid Search for Logistic Regregresion LDA data:")
        best_param_log_lda = grid_search_logreg(X_train_LDA,y_train)

        print("\nGrid Search for KNN balanced data:")
        #best_param_knn_balanced = grid_search_KNN(X_train,y_train)
        print("\n\nGrid Search for KNN PCA data:")
        #best_param_knn_pca = grid_search_KNN(X_train_PCA,y_train)
        print("\n\nGrid Search for KNN LDA data:")
        #best_param_knn_lda = grid_search_KNN(X_train_LDA,y_train)

        print("\n\nBest Parameters for Random forest:")
        print("\nBalanced: ")
        print(best_param_forest_balanced)
        print("\nPCA: ")
        print(best_param_forest_pca)
        print("\nLDA: ")
        print(best_param_forest_lda)

        print("\n\nBest Parameters for Logistic Regresion:")
        print("\nBalanced: ")
        print(best_param_log_balanced)
        print("\nPCA: ")
        print(best_param_log_pca)
        print("\nLDA: ")
        print(best_param_log_lda)

        print("\n\nBest Parameters for KNN:")
        print("\nBalanced: ")
        print(best_param_knn_balanced)
        print("\nPCA: ")
        print(best_param_knn_pca)
        print("\nLDA: ")
        print(best_param_knn_lda)


    if not(no_classify):
        print("add all code for running classification")
        print("\n\nRandom Forest on raw dataset:")
        classification_random_forest(X_train,y_train,X_test,y_test, best_param_forest_balanced)
        print("\n\nRandom Forest on PCA:")
        classification_random_forest(X_train_PCA,y_train,X_test_PCA,y_test, best_param_forest_pca)
        print("\n\nRandom Forest on LDA:")
        classification_random_forest(X_train_LDA,y_train,X_test_LDA,y_test, best_param_forest_lda)
        
        print("\n\nLogistic regression on raw dataset:")
        classification_logistic_regression(X_train,y_train,X_test,y_test, best_param_log_balanced)
        print("\n\nLogistic regression on PCA:")
        classification_logistic_regression(X_train_PCA,y_train,X_test_PCA,y_test, best_param_log_pca)
        print("\n\nLogistic regression on LDA:")
        classification_logistic_regression(X_train_LDA,y_train,X_test_LDA,y_test, best_param_log_lda)
        print("\n\nKNN on raw dataset:")
        
        classification_knn(X_train,y_train,X_test,y_test, best_param_knn_balanced)
        print("\n\nKNN on PCA:")
        classification_knn(X_train_PCA,y_train,X_test_PCA,y_test, best_param_knn_pca)
        print("\n\nKNN on LDA:")
        classification_knn(X_train_LDA,y_train,X_test_LDA,y_test, best_param_knn_lda)
        print("\n")
    
    if not(no_cluster):
        print("add all code for running clustering")
        print("\nKMeans clustering on original data:")
        KMeans_clustering(data.iloc[: , 1:20532],labels['Class'])
        
        print("\n\nKMeans clustering on balanced data:")
        KMeans_clustering(data_balanced.iloc[: , 1:],labels_balanced['Class'], scenario = 'balanced')
        
        print("\n\nKMeans clustering on reduced data:")
        try:
            KMeans_clustering(concatenate((X_train_LDA, X_test_LDA)), concat([y_train,y_test]), scenario = 'reduced')
        except UnboundLocalError:
            print("Data from LDA not existant!!!!!!!!")
            print("\nRunning LDA now")
            X_train_LDA, X_test_LDA = LDA_n_components(X_train, X_test, y_train, y_test, select_n_components_lda(data_balanced.iloc[: , 1:], labels_balanced['Class'], 0.95))
            print("\nRunning Clustering for LDA now")
            KMeans_clustering(concatenate((X_train_LDA, X_test_LDA)), concat([y_train,y_test]), scenario = 'reduced')
    
    if not(no_evaluate):
        print("add all code for running evaluation")


if __name__ == '__main__':

    arguments = get_arguments()
    # To not perform a part of the task mention it in run command line. For example: -no-analysis
    run(
        arguments.no_analysis,
        arguments.no_features,
        arguments.no_classify,
        arguments.no_cluster,
        arguments.no_search,
        arguments.no_evaluate
        )
    

    
