from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation
from pandas import concat
from Code.utils import save_model

def train_baseline(train, augmented=False):
    model = RandomForestClassifier(max_depth=5, max_leaf_nodes=15, n_jobs=-1)
    model.fit(train.loc[:, ~train.columns.isin(["Unnamed: 0", 'Class'])], train["Class"])
    if augmented:
        print("$$$$$$ Trained 'baseline_augmented' model $$$$$$\n\n")
        save_model(model, "baseline_augmented")
    else:
        print("$$$$$$ Trained 'baseline' model $$$$$$\n\n")
        save_model(model, "baseline")

def train_semi_supervised(train_lab, train_unlab):
    model = LabelPropagation(kernel='knn', max_iter=100, n_jobs=-1)
    train_mixed = concat([train_lab, train_unlab])
    model.fit(train_mixed.loc[:, ~train_mixed.columns.isin(["Unnamed: 0", 'Class'])], train_mixed["Class"])
    print("$$$$$$ Trained 'semi_supervised' model $$$$$$\n\n")
    save_model(model, "semi_supervised")

    new_labels=model.transduction_
    train_mixed["Class"] = new_labels
    train_mixed.to_csv('./data/train_mixed_labels.csv')

    return train_mixed