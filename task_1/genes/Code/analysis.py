from Code.utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def plot_classes_histogram_genes(labels):
    print("Label value count is:")
    print(labels.Class.value_counts())
    classes = labels['Class'].tolist()
    plt.hist(classes, bins = 13)
    plt.title("Histogram of class labels")
    plt.ylabel("No. of occurences for each class")
    plt.xlabel("Classes")
    plt.show()

def plot_PCA_components(X_train):
    pca_test = PCA(n_components=30)
    pca_test.fit(X_train)
    sns.set(style='whitegrid')
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.title("PCA test for 'Genes' dataset")
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
    plt.show()

def select_n_components_lda(data, labels, goal_var: float) -> int:
    lda = LinearDiscriminantAnalysis(n_components=None)
    X_lda = lda.fit(data, labels)
    lda_var_ratios = lda.explained_variance_ratio_
    total_variance = 0.0
    n_components = 0

    for explained_variance in lda_var_ratios:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    
    return n_components










