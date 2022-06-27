import random
from pathlib import Path
import math
   
def data_splitter(root: Path, ratio: float = 0.8, val_ratio: float = 0.9):
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

if __name__ == "__main__":
    data_splitter(Path("Dresden/natural"), 0.8)