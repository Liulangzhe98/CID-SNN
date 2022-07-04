import random
from pathlib import Path
import math
import json   

from PIL import Image

def __data_splitter(root: Path, ratio: float = 0.8, val_ratio: float = 0.9):
    random.seed(20052022)  
    train_set = []
    val_set = []
    test_set = []
    total_len = 0
    
    for class_label, device_dir in enumerate(root.glob('*')):
        all_image_paths_per_device = sorted(device_dir.glob('*'))
        random.shuffle(all_image_paths_per_device)
        total_len += len(all_image_paths_per_device)
        num_train_images_per_device = int(len(all_image_paths_per_device) * ratio)
        num_val_images_per_device = int(num_train_images_per_device*val_ratio)
        temp = all_image_paths_per_device[:num_train_images_per_device]
       
        train_set += [(x, class_label) for x in temp[:num_val_images_per_device]]
        val_set   += [(x, class_label) for x in temp[num_val_images_per_device:]]
        test_set  += [(x, class_label) for x in all_image_paths_per_device[num_train_images_per_device:]]
    return train_set, val_set, test_set

def size(root: Path):
    
    for class_label, device_dir in enumerate(root.glob('*')):
        for filename in device_dir.glob("*"):
        
            im = Image.open(filename)
            width, height = im.size
            if width < 800 or height < 800:
                print(f"{filename} | {width} x {height}")
       




def data_splitter(root: Path, ratio: float = 0.8, val_ratio: float = 0.9):
    random.seed(20052022)  
    train_set = []
    val_set = []
    test_set = []
    total_len = 0

    counter = 0
    labels_dict = {}
    bases_dict = {}
    for x in sorted(root.glob("*")):
        if (base := "_".join(str(x).split("_")[:-1])) not in bases_dict.keys():
            # New base model so add base model + full model
            bases_dict[base] = counter
            labels_dict[str(x)] = counter
            counter += 1
        else:
            labels_dict[str(x)] = bases_dict[base]
    

    labels_dict = dict(sorted(labels_dict.items(), key=lambda x: x[0]))

    # print(json.dumps(labels_dict, indent=2))
    
    for device_dir in sorted(root.glob('*')):
        class_label = labels_dict[str(device_dir)]
        all_image_paths_per_device = sorted(device_dir.glob('*'))
        random.shuffle(all_image_paths_per_device)
        total_len += len(all_image_paths_per_device)
        num_train_images_per_device = int(len(all_image_paths_per_device) * ratio)
        num_val_images_per_device = int(num_train_images_per_device*val_ratio)
        temp = all_image_paths_per_device[:num_train_images_per_device]
       
        train_set += [(x, class_label) for x in temp[:num_val_images_per_device]]
        val_set   += [(x, class_label) for x in temp[num_val_images_per_device:]]
        test_set  += [(x, class_label) for x in all_image_paths_per_device[num_train_images_per_device:]]
    # print(test_set[400:600])
    return train_set, val_set, test_set


if __name__ == "__main__":
    #data_splitter(Path("Dresden/natural"), 0.8)
    size(Path("Dresden/natural"))
