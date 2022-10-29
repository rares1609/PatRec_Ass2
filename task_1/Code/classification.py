from Code.utils import *
from Code.features import *

def PCA_random_forest():
    #X_train_scaled, X_test_scaled, y_train = scale_data_genes()
    #pca = PCA(n_components=10)
    #pca.fit(X_train_scaled)
    #X_train_scaled_pca = pca.transform(X_train_scaled)
    #X_test_scaled_pca = pca.transform(X_test_scaled)
    #pca_test_df = pd.DataFrame(data = X_train_scaled_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
    #pca_test_df = pd.concat([pca_test_df, labels['Class']], axis = 1)
    #display(pca_test_df.head(10))
    pca_df, x_train_scaled_pca, y_train = PCA_10_components()
    display(pca_df.head(10))
    print(pca_df.shape)
    rfc = RandomForestClassifier()
    rfc.fit(x_train_scaled_pca, y_train)
    display(rfc.score(x_train_scaled_pca, y_train))
    