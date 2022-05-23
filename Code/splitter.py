from cgi import test
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image
   
def data_splitter(root: Path, ratio: float = 0.8):
    random.seed(20052022)  
    train_set = []
    test_set = []
    # sizes = defaultdict(int)
    

    for class_label, device_dir in enumerate(root.glob('*')):
        all_image_paths_per_device = sorted(device_dir.glob('*'))
        # for x in all_image_paths_per_device:
        #     try:
        #         with Image.open(x) as y:
        #             sizes[y.size] += 1
        #     except:
        #         sizes['UNKNOWN'] += 1


        random.shuffle(all_image_paths_per_device)
        num_train_images_per_device = int(len(all_image_paths_per_device) * ratio)
        train_set += [(x,class_label) for x in all_image_paths_per_device[:num_train_images_per_device]]
        test_set  += [(x,class_label) for x in all_image_paths_per_device[num_train_images_per_device:]]
    # print(sizes)
    return train_set, test_set

if __name__ == "__main__":
    data_splitter(Path("Dresden/natural"), 0.8)