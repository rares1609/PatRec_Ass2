from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation
from pandas import concat
from Code.utils import save_model

# Function that trains baseline (Random Forest) on both augmented and basic train datasets
def train_baseline(train, augmented=False):
    # Declare Model
    model = RandomForestClassifier(max_depth=5, max_leaf_nodes=15, n_jobs=-1)

    # Fit model to train data
    model.fit(train.loc[:, ~train.columns.isin(["Unnamed: 0", 'Class'])], train["Class"])

    # Save model
    if augmented:
        print("$$$$$$ Trained 'baseline_augmented' model $$$$$$\n\n")
        save_model(model, "baseline_augmented")
    else:
        print("$$$$$$ Trained 'baseline' model $$$$$$\n\n")
        save_model(model, "baseline")

# Function that trains semi supervised (Label Propagation based on KNN) on label+unlabeled datasets
def train_semi_supervised(train_lab, train_unlab):
    # Declare Model
    model = LabelPropagation(kernel='knn', max_iter=100, n_jobs=-1)

    # concatenating labeled and unlabeled
    train_mixed = concat([train_lab, train_unlab])

    # Fit model to data
    model.fit(train_mixed.loc[:, ~train_mixed.columns.isin(["Unnamed: 0", 'Class'])], train_mixed["Class"])
    
    # Save Model
    print("$$$$$$ Trained 'semi_supervised' model $$$$$$\n\n")
    save_model(model, "semi_supervised")

    # Extract predicted labels for dataset, by using transduction
    new_labels=model.transduction_

    # Assigning the new labels to the concatenated dataset and saving the data to csv
    train_mixed["Class"] = new_labels
    train_mixed.to_csv('./data/train_mixed_labels.csv')

    # returning the dataset with new labels
    return train_mixed