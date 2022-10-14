from Code.utils import *
import matplotlib.pyplot as plt

def extract_classses_genes():
    _, labels = read_data_genes()
    classes = labels['Class'].tolist()
    return classes

def plot_classes_histogram_genes():
    classes = extract_classses_genes()
    plt.hist(classes, bins = 13)
    plt.title("Histogram of class labels")
    plt.ylabel("No. of occurences for each class")
    plt.xlabel("Classes")
    plt.show()








