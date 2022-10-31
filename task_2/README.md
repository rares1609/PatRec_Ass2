# PatRec_Ass2 Task 2
Assignment 2 Task 2 for Patern Recognition


# Project Structure

The project contains a "run.py" script for running the code, a "requirements.txt" file, "data" folder where the data necessary for assignment needs to be "Code" folder with all the aditional scripts necessary for running the code

- **run.py** : script that needs to be run to run the entire task

## Code Folder

Includes:
- **test.py** : script containing all the function necessary for testing the models. Used in **run.py**

- **train.py** : script containing all the function necessary for training the models. Used in **run.py**

- **utils.py** : script including utilitary functions


## data Folder

Should be included:
- **creditcard.csv** : includes all data necessary for running the task

## Requirements

joblib==1.2.0

matplotlib==3.6.1

numpy==1.23.3

pandas==1.5.0

scikit_learn==1.1.3

tqdm==4.64.1

## The Report
report is given as a pdf file

# Running Task

## Prerequisit:

pip install -r requirements.txt

## Running task

1. Either directly run the script "run.py" or use command "python run.py" from the working directory

2. To exclude different parts of the tasks use function call arguments:

**-no-test** : the testing part of the task will not run
**-no-train** : the training part of the task will not run (has to be run at least once)
**-runs= no_runs** : tell the script how many runs to perform, input needs to be integer (default is 1)

# Output

## data Folder

Data split into labeled, unlabeled and test are also saved in this folder. In adition the dataset with the predicted labels from semi supervised learning is also saved here

## models Folder

All trained models are saved with coresponding names in this folder

## results Folder

For more that 1 run plots are created for accuracy, jaccard and f1 score for all models in all cases and are saved in this folder.

# Copyright
Scripts created by:

Paul Pintea (s3593673) && Rares Adrian Oancea (s3974537) && Anthony Klinker (s3513556)

