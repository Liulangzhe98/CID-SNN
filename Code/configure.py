import torch
import torchvision.transforms as tf

import json


class Configure(object):
    '''Class that keeps all the configurable parts'''

    # Constants that are the same for local and peregrine mode
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    HYPER = {
        "LR":  0.0002,
        "DECAY": 0.90
    }

    possible_transforms = {
        "small":  tf.Compose([
            tf.CenterCrop(100),
            tf.ToTensor()
        ]),
        "medium": tf.Compose([
            tf.CenterCrop(200),
            tf.ToTensor()
        ]),
        "large":  tf.Compose([
            tf.CenterCrop(800),
            tf.ToTensor()
        ]),
        "guru": tf.Compose([
            tf.CenterCrop(800),
            tf.ToTensor()
        ])
    }

    def __init__(self, args) -> None:
        self.CROP_SIZE = getattr(args, 'size')
        self.LOAD_ID = getattr(args, 'ID')[0]
        self.Transform = self.possible_transforms[self.CROP_SIZE]

        if getattr(args, 'dev'):  # Local settings
            print("\u001b[31m == RUNNING IN DEV MODE == \u001b[0m")
            print(args)

            self.MAX_EPOCH = 15
            self.SUBSET_SIZE = 0.005
            self.SUBSET_VAL_SIZE = 0.005

            self.DATA_FOLDER = "Dresden/natural"
            self.RESULT_FOLDER = "Results"
            self.MODELS_FOLDER = "Models"
            self.SAVED_EXPERIMENTS = "experiments.json"
        else:  # Peregrine settings
            print(" == RUNNING IN PEREGRINE MODE == ")
            self.MAX_EPOCH = 50
            self.SUBSET_SIZE = 1.0
            self.SUBSET_VAL_SIZE = 1.0

            self.DATA_FOLDER = "Project/Dresden/natural"
            self.RESULT_FOLDER = "Project/Results"
            self.MODELS_FOLDER = "Project/Models"
            self.SAVED_EXPERIMENTS = "Project/experiments.json"

        # Read previously done experiments file and create a new key for the model
        with open(self.SAVED_EXPERIMENTS, "r") as f:
            try:
                saved_object = json.load(f)
                exp_id = len(list(filter(
                    lambda x: x.startswith(self.CROP_SIZE),
                    saved_object.keys())))
            except json.JSONDecodeError:
                exp_id = 0
        
        self.EXP_ID = f"{self.CROP_SIZE}_{exp_id}"
        
        # Storing the loaded model filename so it can be used later on
        if (suffix := getattr(args, 'load')):  # There is a string after the --load
            load_name = f"{self.CROP_SIZE}_{suffix}"
            if load_name not in saved_object.keys():
                print(f"Could not load the requested model: {load_name}")
                exit(1)
            self.loaded_model = saved_object[load_name]['Model']

        # File names
        self.SAVE_MODEL_AS = f"{self.MODELS_FOLDER}/model_{self.EXP_ID}.pth"
        self.LOSS_GRAPH = f'{self.RESULT_FOLDER}/loss_graph_{self.LOAD_ID}.png'
        self.MULTIPLE = f"{self.RESULT_FOLDER}/histo_multiple_{self.LOAD_ID}.svg"
        self.SUMMED = f"{self.RESULT_FOLDER}/histo_summed_{self.LOAD_ID}.svg"


    def __str__(self) -> str:
        return f""
