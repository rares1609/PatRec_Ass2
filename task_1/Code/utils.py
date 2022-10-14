from genericpath import isdir
from os import listdir, mkdir, path, remove
from pathlib import Path
from PIL import Image
import tqdm
import imagehash
import pandas as pd

def read_data_genes(datapath = "./data/Genes/data.csv", labelpath = "./data/Genes/labels.csv"):
    data = pd.read_csv(datapath)
    labels = pd.read_csv(labelpath)
    return data, labels

def read_data_cats(path="./data/BigCats"):
    if not(Path("./data/BigCats").exists()):
        mkdir("./data")
        print("ADD BigCats dataset to 'data' folder!!!!!!!!!!!!!")
    classes = listdir(path)
    img_dict = {}
    for folder in classes:
        img_dict[folder] = []
        for image_name in listdir(path + "/" + folder):
            img = Image.open(path + "/" + folder + "/" + image_name)
            img_dict[folder].append(img)
    return classes, img_dict

def check_data_directory(name):
    path = "./data"
    name = name.split("/")
    for item in name:
        path = Path(str(path) + "/" + item)
        if not(path.exists()):
            mkdir(path)
            return False
    return True

def check_image_similarity(img1, img2):
    hash0 = imagehash.average_hash(Image.open(img1)) 
    hash1 = imagehash.average_hash(Image.open(img2)) 
    cutoff = 5  # maximum bits that could be different between the hashes. 

    if hash0 - hash1 < cutoff:
        return True
    else:
        return False


def remove_dupicate_images(path):
    if len(listdir(path)) < 40:
        print("Augumentation Process, checking and deleting duplicates:")
        for img1 in listdir(path):
            for img2 in listdir(path):
                if img1 != img2:
                    if check_image_similarity(path + "/" + img1, path + "/" + img2):
                        try:
                            remove(path + "/" + img1)
                        except FileNotFoundError:
                            pass #file already deleted
                        break