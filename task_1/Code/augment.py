import shutil
from Code.utils import *

def equalize_cats_classes(img_dict):
    check_data_directory("augmented_cats/equalized")
    path = "./data/augmented_cats/equalized/"
    for folder_name in img_dict.keys():
        try:
            shutil.copytree("./data/Bigcats/" + folder_name, path + folder_name)
            remove_dupicate_images(path + folder_name)
            mirror_images(path + folder_name)
        except FileExistsError:
            mirror_images(path + folder_name)
            pass
        