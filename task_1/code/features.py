from Code.utils import *
from Code.analysis import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def generate_features_genes():
    features = []
    for i in range(0, 20531):
        features.append('gene_'+str(i))
    return features

def standardize_data_genes():
    data, labels = read_data_genes()
    features = generate_features_genes()
    data2 = data.assign(Class = labels['Class'])
    x = data2.loc[:, features].values
    y = data2.loc[:,['Class']].values
    x = StandardScaler().fit_transform(x)
    return x, y

def PCA_Projection_genes():
    data, labels = read_data_genes()
    data2 = data.assign(Class = labels['Class'])
    x, y = standardize_data_genes()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, data2[['Class']]], axis = 1)
    return finalDf

def plot_projection_genes():
    df = PCA_Projection_genes()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['PRAD', 'LUAD', 'BRCA', 'KIRC', 'COAD']
    colors = ['r', 'g', 'b', 'y', 'm']
    for target, color in zip(targets,colors):
        indicesToKeep = df['Class'] == target
        ax.scatter(df.loc[indicesToKeep, 'principal component 1']
               , df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()








