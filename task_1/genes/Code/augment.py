from sklearn.utils import resample
from pandas import DataFrame, concat

def resample_data(data, labels):
    data['Class'] = labels['Class']
    del labels
    val = data['Class'].value_counts().values.tolist()
    BRCA = data[data.Class=='BRCA']
    KIRC = resample(data[data.Class=='KIRC'], 
                            replace=True,
                                n_samples=val[0],
                                    random_state=12)
    LUAD = resample(data[data.Class=='LUAD'], 
                            replace=True,
                                n_samples=val[0],
                                    random_state=12)
    PRAD = resample(data[data.Class=='PRAD'], 
                            replace=True,
                                n_samples=val[0],
                                    random_state=12)
    COAD = resample(data[data.Class=='COAD'], 
                            replace=True,
                                n_samples=val[0],
                                    random_state=12)
    data = concat([BRCA, KIRC, LUAD, PRAD, COAD])
    labels = DataFrame()
    labels['Class'] = data['Class']
    data.drop('Class', inplace=True, axis=1)

    return data,labels