import pandas as pd
from os import mkdir
from pathlib import Path
from joblib import dump,load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def read_data():
    if not(Path("./data/test.csv").exists()) or not(Path("./data/train_lab.csv").exists()) or not(Path("./data/train_unlab.csv").exists()):
        data = pd.read_csv('./data/creditcard.csv')
        data.drop('Time', inplace=True, axis=1)
        data.drop('Amount', inplace=True, axis=1)
    
        train, test = train_test_split(data, test_size=0.2)
        print(train)
        train_lab, train_unlab = train_test_split(train, test_size=0.7)
        train_unlab['Class'] = -1
    
        test.to_csv('./data/test.csv')
        train.to_csv('./data/train.csv')
        train_lab.to_csv('./data/train_lab.csv')
        train_unlab.to_csv('./data/train_unlab.csv')
    
    else:
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