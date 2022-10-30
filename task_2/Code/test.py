from numpy import mean
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, balanced_accuracy_score, classification_report
from Code.utils import load_model

def test_model(model_name: str, test):
    print("$$$$$$ Testing " + model_name + " $$$$$$")
    model = load_model(model_name)
    predictions = model.predict(test.loc[:, ~test.columns.isin(["Unnamed: 0", 'Class'])])
    acc = accuracy_score(test["Class"], predictions)
    jac = jaccard_score(test["Class"], predictions)
    f1 = f1_score(test["Class"], predictions)
    print("The results of the '" + model_name + "' are:\nAccuracy: " + str(100 * acc) + "%\nF1-score: " + str(f1) +"\nJaccard Score: " +  str(jac) + "\n\n")
    #print(balanced_accuracy_score(test["Class"], predictions))
    #print(classification_report(test["Class"], predictions, target_names=["Class 0", "Class 1"]))