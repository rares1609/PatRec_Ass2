from Code.utils import *
from Code.analysis import *
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
def PCA_10_components(X_train, X_test):
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(X_train_pca)
    #pca_df = pd.DataFrame(data = X_train_scaled_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
    #pca_df = pd.concat([pca_df, labels['Class']], axis = 1)
    return X_train_pca, X_test_pca

def LDA_n_components(X_train, X_test, y_train, y_test, n):
    lda = LinearDiscriminantAnalysis(n_components=n)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    return X_train, X_test






