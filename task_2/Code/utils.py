import pandas as pd
from os import mkdir
from pathlib import Path
from joblib import dump,load
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def balance_data(data):
    indeces = data['Class'].value_counts().index.tolist()
    val = data['Class'].value_counts().values.tolist()
    if val[0] != val[1]:
        print("Data is unbalanced")
        print(data.Class.value_counts())
        data_majority = data[data.Class == 0]
        data_minority = data[data.Class == 1]
        majority_downsampled = resample(data_majority,
                                            replace=False,
                                                n_samples=int(val[0]/2), # making minority match the majority
                                                    random_state=12) # sampling in the same way for reproductibility
        minority_upsampled = resample(data_minority,
                                        replace=True,
                                            n_samples=int(val[0]/2), # making minority match the majority
                                                random_state=12) # sampling in the same way for reproductibility
        
        balanced = pd.concat([majority_downsampled, minority_upsampled])
        print("The new value counts are:")
        print(balanced.Class.value_counts())
    
    else:
        print("Data is balanced")
        print(data.Class.value_counts())
        balanced = data

    return balanced


def read_data():
    if not(Path("./data/test.csv").exists()) or not(Path("./data/train_lab.csv").exists()) or not(Path("./data/train_unlab.csv").exists()):
        print("Extracting data\n")
        data = pd.read_csv('./data/creditcard.csv')
        data.drop('Time', inplace=True, axis=1)
        data.drop('Amount', inplace=True, axis=1)

        data = balance_data(data)
    
        train, test = train_test_split(data, test_size=0.2)
        train_lab, train_unlab = train_test_split(train, test_size=0.7)
        train_unlab['Class'] = -1
    
        test.to_csv('./data/test.csv')
        train.to_csv('./data/train.csv')
        train_lab.to_csv('./data/train_lab.csv')
        train_unlab.to_csv('./data/train_unlab.csv')
    
    else:
        print("Loading data\n")
        test = pd.read_csv('./data/test.csv')
        train_lab = pd.read_csv('./data/train_lab.csv')
        train_unlab = pd.read_csv('./data/train_unlab.csv')

    return train_lab, train_unlab, test


def save_model(model, name="baseline"):
    if not(Path("./models").exists()):
        mkdir("./models")
    dump(model, "./models/" + name + ".joblib")

def load_model(name="baseline"):
    if Path("./models/" + name + ".joblib").exists():
        model = load("./models/" + name + ".joblib")
    else:
        print("Model "+ name + " does not exist first train this model")
    return model