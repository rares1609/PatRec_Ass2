from numpy import mean
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, balanced_accuracy_score, classification_report
from Code.utils import load_model

# function that tests each model
def test_model(model_name: str, test):
    print("$$$$$$ Testing " + model_name + " $$$$$$")

    # Import coresponding model
    model = load_model(model_name)

    # Make predictions
    predictions = model.predict(test.loc[:, ~test.columns.isin(["Unnamed: 0", 'Class'])])

    # Calculate the accuracy jaccard score and f1 score
    acc = accuracy_score(test["Class"], predictions)
    jac = jaccard_score(test["Class"], predictions)
    f1 = f1_score(test["Class"], predictions)

    # printing results for the model
    print("The results of the '" + model_name + "' are:\nAccuracy: " + str(100 * acc) + "%\nF1-score: " + str(f1) +"\nJaccard Score: " +  str(jac) + "\n\n")

    return acc, jac, f1