# Imports
from pathlib import PosixPath
from numpy import diff
from torch import tensor
from torch.utils.data import DataLoader

import time

from SNN import *
from helper import *
from splitter import *

def validate_model(net: SiameseNetwork, dataloader: DataLoader, image_set: DresdenSNNDataset,  config):
     
    print(" === Validation results === ")

    net.train(False)
    with torch.no_grad():
        device = config["DEVICE"]
        start_time = time.time()
        models_checked = {}
        cameras = set([(str(x), label) for (x, label) in image_set.image_paths])
        pre_selected = random.sample(cameras, k=9)
        for x, truth in pre_selected:
            base_image = image_set.images[PosixPath(x)]
            base_image = base_image[None, :]
            same_histo = []
            diff_histo = []
            for (y, label) in image_set.image_paths:
    
                compare_image = image_set.images[PosixPath(y)]
                compare_image = compare_image[None, :]
            
                pred = net(base_image.to(device), compare_image.to(device))
    
                # print(pred.item())
                difference = pred.item()
                # print(f"{'same' if truth == label else 'diff'} | expect: {int(truth==label)} vs real: {difference:.4f}")
                if truth == label:
                    same_histo.append(difference)
                else:
                    diff_histo.append(difference)
            

            
            models_checked[x] = {
                "same" : same_histo,
                "diff" : diff_histo
            }
            print(f"  Validated : {x:<30} | {time.time()-start_time:>5.2f}s")
            start_time = time.time()
                
        models_checked["Summed"] = {
            "same" : [x for (_, v) in models_checked.items() for x in v['same']],
            "diff" : [x for (_, v) in models_checked.items() for x in v['diff']]
        }

    multiple_histo(models_checked, config["RESULT_FOLDER"]+f"/histo_multiple_new.png")

def validate_model_with_loader(net: SiameseNetwork, dataloader: DataLoader, config):
    print(" === Validation results === ")
    device = config["DEVICE"]
    dataiter = iter(dataloader)
    net.train(False)
    with torch.no_grad():
        x0, _, _, f_name0, _ = next(dataiter)
        name = f_name0[0].split('/')[0]

        models_checked = {}
        for i in range(8):
            same_histo = []
            diff_histo = []

            while name in models_checked.keys():
                x0, _, _, f_name0, _ = next(dataiter)
                name = f_name0[0].split('/')[0]
            
            dataiter = iter(dataloader)
            print(f"  Validating on : {name}")
            
            for _, x1, _, _, f_name1 in dataiter:
                name_other = f_name1[0].split('/')[0]
                output1 = net(x0.to(device), x1.to(device))
                
                # print(output1)

                if name == name_other:
                    same_histo += output1.tolist()
                else:
                    diff_histo += output1.tolist()
            models_checked[name] = {
                "same" : np.array(same_histo).flatten(),
                "diff" : np.array(diff_histo).flatten()
            }
            dataiter = iter(dataloader)

            models_checked["Summed"] = {
                "same" : [x for (_, v) in models_checked.items() for x in v['same']],
                "diff" : [x for (_, v) in models_checked.items() for x in v['diff']]
            }

            # histo_makers(same_histo, diff_histo, name, config["RESULT_FOLDER"]+ "/histo_together.png")  
    multiple_histo(models_checked, config["RESULT_FOLDER"]+f"/histo_multiple_new_loader.png")