import argparse
import sys
from tqdm import tqdm
from Code.train import *
from Code.test import *
from Code.utils import *

# function that determines the function call parameters and returns them
def get_arguments():
    parser = argparse.ArgumentParser(description='Pattern Recogniton Task1')

    # arguments allowed in function call
    parser.add_argument('-no-train', action=argparse.BooleanOptionalAction, help='Do not Perform Training')
    parser.add_argument('-no-test', action=argparse.BooleanOptionalAction, help='Do not Perform Testing')
    parser.add_argument('-runs', type=int, default=1, help='How many times the program is executed')

    
    arguments = parser.parse_args()
    # returning each function call parameter
    return arguments

# Function that runs entire code
def run(no_train, no_test, runs):
    # reading the split data
    data_train_lab, data_train_unlab, data_test = read_data()
    # declaring results lists
    baseline_acc, baseline_jac, baseline_f1, semi_supervised_acc, semi_supervised_jac, semi_supervised_f1, augmented_acc, augmented_jac, augmented_f1 = initialize_lists()
    
    # running for specified number of runs
    for i in tqdm(range(runs)):
        
        # block of code for running training
        if not(no_train):
            train_baseline(data_train_lab)
            data_train_mixed = train_semi_supervised(data_train_lab, data_train_unlab)
            train_baseline(data_train_mixed, augmented=True)

        # block of code for running testing
        if not(no_test):
            if use_model == "all":
                base_acc, base_jac, base_f1 = test_model("baseline", data_test)
                semi_acc, semi_jac, semi_f1 = test_model("semi_supervised", data_test)
                aug_acc, aug_jac, aug_f1= test_model("baseline_augmented", data_test)

        # appending acc jaccard and f1 score for each specific model to their coresponding list
        baseline_acc.append(base_acc)
        baseline_jac.append(base_jac)
        baseline_f1.append(base_f1)
        semi_supervised_acc.append(semi_acc)
        semi_supervised_jac.append(semi_jac)
        semi_supervised_f1.append(semi_f1)
        augmented_acc.append(aug_acc)
        augmented_jac.append(aug_jac)
        augmented_f1.append(aug_f1)

    # If there is more than 1 run make and save plots for all accuracies, jaccard and f1 scores
    if runs > 1:
        plot_results(runs, baseline_acc, baseline_jac, baseline_f1, semi_supervised_acc, semi_supervised_jac, semi_supervised_f1, augmented_acc, augmented_jac, augmented_f1)


if __name__ == '__main__':

    arguments = get_arguments()
    # To not perform a part of the task mention it in run command line. For example: -no-training
    run(
        arguments.no_train,
        arguments.no_test,
        arguments.runs
        )