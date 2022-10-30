import argparse
import sys
from Code.analysis import *
from Code.features import *
from Code.classification import *
from Code.features import *
from Code.classification import *
from Code.clustering import *

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
    if not(no_analysis):
        print("add all code for running analysis")
        plot_classes_histogram_genes()

    if not(no_features):
        print("add all code for running feature extraction")
        #plot_2D_PCA()
        PCA_test()
        baseline_random_forest()
        

    if not(no_classify):
        print("add all code for running classification")
        #forest_test()
        #random_forest()
        PCA_random_forest()
    
    if not(no_cluster):
        print("add all code for running clustering")
        clustering()
    
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
    

    
