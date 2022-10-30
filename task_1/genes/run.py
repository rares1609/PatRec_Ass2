import argparse
import sys
from Code.analysis import plot_classes_histogram_genes,plot_PCA_components,select_n_components_lda
from Code.features import *
from Code.classification import *
from Code.clustering import *
from Code.augment import *

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

def run(no_analysis, no_features, no_classify, no_cluster, no_search, no_evaluate):
    data, labels = read_data_genes()
    data_balanced, labels_balanced = resample_data(data, labels)
    X_train, X_test, y_train, y_test = data_split_genes(data_balanced, labels_balanced)
    if not(no_analysis):
        print("add all code for running analysis")
        #plot_classes_histogram_genes(labels)
        #plot_classes_histogram_genes(labels_balanced)
        #plot_PCA_components(X_train)

    if not(no_features):
        print("add all code for running feature extraction")
        X_train_PCA, X_test_PCA = PCA_10_components(X_train, X_test)
        print("Number of components for LDA: ", select_n_components_lda(data_balanced.iloc[: , 1:], labels_balanced['Class'], 0.95))
        X_train_LDA, X_test_LDA = LDA_n_components(X_train, X_test, y_train, y_test, select_n_components_lda(data_balanced.iloc[: , 1:], labels_balanced['Class'], 0.95))
 
        

    if not(no_classify):
        classification_random_forest(X_train,y_train,X_test,y_test)
        classification_random_forest(X_train_PCA,y_train,X_test_PCA,y_test)
        classification_random_forest(X_train_LDA,y_train,X_test_LDA,y_test)
        print("add all code for running classification")
    
    if not(no_cluster):
        print("add all code for running clustering")
    
    if not(no_search):
        print("add all code for running grid search")
    
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
    

    
