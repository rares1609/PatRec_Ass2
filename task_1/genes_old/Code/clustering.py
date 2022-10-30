from Code.utils import *
from Code.features import *
from Code.classification import *
from sklearn.cluster import DBSCAN

def clustering():
    pca_df, x_train_scaled_pca, y_train = PCA_10_components()
    clt = DBSCAN()
    model = clt.fit(x_train_scaled_pca)
    clusters = pd.DataFrame(model.fit_predict(x_train_scaled_pca))
    pca_df["Clusters"] = clusters
    fig = plt.figure(figsize=(10,10)); ax = fig.add_subplot(111)
    scatter = ax.scatter(pca_df[0],pca_df[1], pca_df[2], pca_df[3], pca_df[4], pca_df[5], pca_df[6], pca_df[7], pca_df[8], pca_df[9],c=pca_df["Cluster"],s=50)
    ax.set_title("DBSCAN Clustering")
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    plt.colorbar(scatter)
    plt.show()
    