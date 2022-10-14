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

def plot_histogram_from_image_dict(img_dict):
    classes_len = []
    for key in img_dict.keys():
        classes_len.append(len(img_dict[key]))
    plt.bar(img_dict.keys(), classes_len)
    plt.ylabel("No. images")
    plt.xlabel("Classes")
    plt.show()






