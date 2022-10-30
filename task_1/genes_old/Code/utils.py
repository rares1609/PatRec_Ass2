import pandas as pd

def read_data_genes(datapath = "./data/Genes/data.csv", labelpath = "./data/Genes/labels.csv"):
    data = pd.read_csv(datapath)
    labels = pd.read_csv(labelpath)
    return data, labels