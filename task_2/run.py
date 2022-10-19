import argparse
import sys
from Code.train import *
from Code.test import *

def get_arguments():
    parser = argparse.ArgumentParser(description='Pattern Recogniton Task1')

    parser.add_argument('-no-train', action=argparse.BooleanOptionalAction, help='Do not Perform Training')
    parser.add_argument('-no-test', action=argparse.BooleanOptionalAction, help='Do not Perform Testing')
    parser.add_argument('-use-saved-model', action=argparse.BooleanOptionalAction, help='Use saved model for testing')


    parser.add_argument('-no-cluster', action=argparse.BooleanOptionalAction, help='Do not Perform Clustering')
    parser.add_argument('-no-search', action=argparse.BooleanOptionalAction, help='Do not Perform Grid Search')
    parser.add_argument('-no-evaluate', action=argparse.BooleanOptionalAction, help='DO not Perform Evaluation')

    
    arguments = parser.parse_args()
    return arguments

def run(no_analysis, no_features, no_classify, no_cluster, no_search, no_evaluate):
    if not(no_analysis):
        print("add all code for training")
        classes, imgs = read_data_cats()
        #plot_histogram_from_image_dict(imgs)
        plot_classes_histogram_genes()

    if not(no_features):
        print("add all code for testing")


if __name__ == '__main__':

    arguments = get_arguments()
    # To not perform a part of the task mention it in run command line. For example: -no-analysis
    run(
        arguments.no_train,
        arguments.no_test,
        arguments.use_saved_model,
        arguments.no_cluster,
        arguments.no_search,
        arguments.no_evaluate
        )