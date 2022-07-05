import random
from pathlib import Path
from typing import Tuple


def __data_splitter(root: Path, ratio: float = 0.8,
                    val_ratio: float = 0.9) -> Tuple[list, list, list]:
    """Deprecated data splitter used for device level splitting

    Args:
        root (Path): The path to the rootfolder (e.g. Dresden/natural). 
        ratio (float): The ratio between train and test set. Defaults to 0.8
        val_ratio (float): The ration between train and validation set. Defaults to 0.9
    Returns:
        tuple(list, list, list): Return a tuple containg the sets of image paths.
    """

    random.seed(20052022)
    train_set = []
    valid_set = []
    test_set = []
    total_len = 0

    # Split all devices into the 3 seperate sets and give them their device label
    for label, device_dir in enumerate(root.glob('*')):
        # aippd = all_images_paths_per_device
        aippd = sorted(device_dir.glob('*'))
        random.shuffle(aippd)
        total_len += len(aippd)
        num_train_images_per_device = int(
            len(aippd) * ratio)
        num_val_images_per_device = int(num_train_images_per_device*val_ratio)
        temp = aippd[:num_train_images_per_device]

        train_set += [(x, label) for x in temp[:num_val_images_per_device]]
        valid_set += [(x, label) for x in temp[num_val_images_per_device:]]
        test_set += [(x, label) for x in aippd[num_train_images_per_device:]]
    return train_set, valid_set, test_set


def data_splitter(root: Path, ratio: float = 0.8,
                  val_ratio: float = 0.9) -> Tuple[list, list, list] :
    """Data splitter used for model level splitting

    Args:
        root (Path): The path to the rootfolder (e.g. Dresden/natural). 
        ratio (float): The ratio between train and test set. Defaults to 0.8
        val_ratio (float): The ration between train and validation set. Defaults to 0.9
    Returns:
        tuple(list, list, list): Return a tuple containg the sets of image paths.
    """
    random.seed(20052022)
    train_set = []
    valid_set = []
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

    # Split all devices into the 3 seperate sets and give them their model label
    for device_dir in sorted(root.glob('*')):
        label = labels_dict[str(device_dir)]
        # aippd = all_images_paths_per_device
        aippd = sorted(device_dir.glob('*'))
        random.shuffle(aippd)
        total_len += len(aippd)
        num_train_images_per_device = int(
            len(aippd) * ratio)
        num_val_images_per_device = int(num_train_images_per_device*val_ratio)
        temp = aippd[:num_train_images_per_device]

        train_set += [(x, label) for x in temp[:num_val_images_per_device]]
        valid_set += [(x, label) for x in temp[num_val_images_per_device:]]
        test_set += [(x, label) for x in aippd[num_train_images_per_device:]]
    return train_set, valid_set, test_set


if __name__ == "__main__":
    data_splitter(Path("Dresden/natural"), 0.8)
