import pandas as pd
from os import mkdir
from statistics import mean
from pathlib import Path
from joblib import dump,load
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# function that balances data
def balance_data(data):
    #get info about how many elements in each class
    indeces = data['Class'].value_counts().index.tolist()
    val = data['Class'].value_counts().values.tolist()

    # if to classes have different no of elements balance data
    if val[0] != val[1]:
        print("Data is unbalanced")
        print(data.Class.value_counts())

        # splitting the two classes 
        data_majority = data[data.Class == 0]
        data_minority = data[data.Class == 1]

        # downsampling majority class to half the size
        majority_downsampled = resample(data_majority,
                                            replace=False,
                                                n_samples=int(val[0]/2), # making minority match the majority
                                                    random_state=12) # sampling in the same way for reproductibility
        
        # upsampling minority class to half of the majority class
        minority_upsampled = resample(data_minority,
                                        replace=True,
                                            n_samples=int(val[0]/2), # making minority match the majority
                                                random_state=12) # sampling in the same way for reproductibility
        
        # concatenating the two classes 
        balanced = pd.concat([majority_downsampled, minority_upsampled])
        print("The new value counts are:")
        print(balanced.Class.value_counts())
    # if data is balanced leave it alone
    else:
        print("Data is balanced")
        print(data.Class.value_counts())
        balanced = data

    # return balanced data
    return balanced

# Function that reads the data and splits it as per the requirements of the assignment
def read_data():
    # if the split files do not exist read/split data
    if not(Path("./data/test.csv").exists()) or not(Path("./data/train_lab.csv").exists()) or not(Path("./data/train_unlab.csv").exists()):
        # read data
        print("Extracting data\n")
        data = pd.read_csv('./data/creditcard.csv')
        # drop not needed columns
        data.drop('Time', inplace=True, axis=1)
        data.drop('Amount', inplace=True, axis=1)

        # balance data
        data = balance_data(data)

        # split data
        train, test = train_test_split(data, test_size=0.2)
        train_lab, train_unlab = train_test_split(train, test_size=0.7)
        train_unlab['Class'] = -1

        # save split data to csv
        test.to_csv('./data/test.csv')
        train.to_csv('./data/train.csv')
        train_lab.to_csv('./data/train_lab.csv')
        train_unlab.to_csv('./data/train_unlab.csv')
    
    # if the data was split before import existing data
    else:
        print("Loading data\n")
        test = pd.read_csv('./data/test.csv')
        train_lab = pd.read_csv('./data/train_lab.csv')
        train_unlab = pd.read_csv('./data/train_unlab.csv')

    # return split data
    return train_lab, train_unlab, test

# Function to save each individual model
def save_model(model, name="baseline"):
    if not(Path("./models").exists()):
        mkdir("./models")
    dump(model, "./models/" + name + ".joblib")

# Function to load each model
def load_model(name="baseline"):
    if Path("./models/" + name + ".joblib").exists():
        model = load("./models/" + name + ".joblib")
    else:
        print("Model "+ name + " does not exist first train this model")
    return model

# Utility function to initialize all results lists as lists
def initialize_lists():
    baseline_acc = []
    baseline_jac = []
    baseline_f1 = []
    semi_supervised_acc = []
    semi_supervised_jac = []
    semi_supervised_f1 = []
    augmented_acc = []
    augmented_jac = []
    augmented_f1 = []
    return baseline_acc, baseline_jac, baseline_f1, semi_supervised_acc, semi_supervised_jac, semi_supervised_f1, augmented_acc, augmented_jac, augmented_f1

# Plotting the results for all the runs 
def plot_results(runs, baseline_acc, baseline_jac, baseline_f1, semi_supervised_acc, semi_supervised_jac, semi_supervised_f1, augmented_acc, augmented_jac, augmented_f1):
    if not(Path('./results').exists()):
        mkdir('./results')
    x = list(range(1, runs+1))
    print(baseline_acc)
    
    #Plotting lines
    plt.plot(x, baseline_acc, label = "Baseline")
    plt.plot(x, semi_supervised_acc, label = "Semi-Supervised")
    plt.plot(x, augmented_acc, label = "Augmented-Baseline")
  
    # naming the x axis
    plt.xlabel('Runs')
    # naming the y axis
    plt.ylabel('Accuracy')
    # giving a title to my graph
    plt.title(f'Accuracy over {runs} runs')
    # show a legend on the plot
    plt.legend()

    plt.savefig(f'./results/accuracy_{runs}runs.png')
    # function to show the plot
    plt.show()

    #Plotting lines
    plt.plot(x, baseline_jac, label = "Baseline")
    plt.plot(x, semi_supervised_jac, label = "Semi-Supervised")
    plt.plot(x, augmented_jac, label = "Augmented-Baseline")
  
    # naming the x axis
    plt.xlabel('Runs')
    # naming the y axis
    plt.ylabel('Jaccard Score')
    # giving a title to my graph
    plt.title(f'Jaccard Score over {runs} runs')
    # show a legend on the plot
    plt.legend()
    plt.savefig(f'./results/jaccard_{runs}runs.png')
    # function to show the plot
    plt.show()

    #Plotting lines
    plt.plot(x, baseline_f1, label = "Baseline")
    plt.plot(x, semi_supervised_f1, label = "Semi-Supervised")
    plt.plot(x, augmented_f1, label = "Augmented-Baseline")
  
    # naming the x axis
    plt.xlabel('Runs')
    # naming the y axis
    plt.ylabel('F1-Score')
    # giving a title to my graph
    plt.title(f'F1-Score over {runs} runs')
    # show a legend on the plot
    plt.legend()
    plt.savefig(f'./results/f1_{runs}runs.png')
    # function to show the plot
    plt.show()

    # Printing the mean results
    print("\n\n$$$$$$ Mean baseline accuracy $$$$$$")
    print(mean(baseline_acc))
    print("\n\n$$$$$$ Mean baseline jaccard $$$$$$")
    print(mean(baseline_jac))
    print("\n\n$$$$$$ Mean baseline f1 $$$$$$")
    print(mean(baseline_f1))
    print("\n\n$$$$$$ Mean semi supervised accuracy $$$$$$")
    print(mean(semi_supervised_acc))
    print("\n\n$$$$$$ Mean semi supervised jaccard $$$$$$")
    print(mean(semi_supervised_jac))
    print("\n\n$$$$$$ Mean semi supervised f1 $$$$$$")
    print(mean(semi_supervised_f1))
    print("\n\n$$$$$$ Mean baseline augmented accuracy $$$$$$")
    print(mean(augmented_acc))
    print("\n\n$$$$$$ Mean baseline augmented jaccard $$$$$$")
    print(mean(augmented_jac))
    print("\n\n$$$$$$ Mean baseline augmented f1 $$$$$$")
    print(mean(augmented_f1))
