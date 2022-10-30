import argparse
import sys
from Code.train import *
from Code.test import *
from Code.utils import *

def get_arguments():
    parser = argparse.ArgumentParser(description='Pattern Recogniton Task1')

    parser.add_argument('-no-train', action=argparse.BooleanOptionalAction, help='Do not Perform Training')
    parser.add_argument('-no-test', action=argparse.BooleanOptionalAction, help='Do not Perform Testing')
    parser.add_argument('-model', type=str, default='all', help='For what model to run testing and training')
    parser.add_argument('-runs', type=int, default=1, help='How many times the program is executed')

    parser.add_argument('-no-cluster', action=argparse.BooleanOptionalAction, help='Do not Perform Clustering')
    parser.add_argument('-no-search', action=argparse.BooleanOptionalAction, help='Do not Perform Grid Search')
    parser.add_argument('-no-evaluate', action=argparse.BooleanOptionalAction, help='DO not Perform Evaluation')

    
    arguments = parser.parse_args()
    return arguments

def run(no_train, no_test, use_model, runs):
    data_train_lab, data_train_unlab, data_test = read_data()
    for i in range(runs):
        if not(no_train):
            if use_model == "all":
                train_baseline(data_train_lab)
                data_train_mixed = train_semi_supervised(data_train_lab, data_train_unlab)
                train_baseline(data_train_mixed, augmented=True)

        if not(no_test):
            if use_model == "all":
                test_model("baseline", data_test)
                test_model("semi_supervised", data_test)
                test_model("baseline_augmented", data_test)


if __name__ == '__main__':

    arguments = get_arguments()
    # To not perform a part of the task mention it in run command line. For example: -no-training
    run(
        arguments.no_train,
        arguments.no_test,
        arguments.model,
        arguments.runs
        )