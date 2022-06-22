# Imports
from pathlib import PosixPath
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

import time

from SNN import *
from helper import *
from splitter import *

def validate_model(net: SiameseNetwork, dataloader: DataLoader, image_set: DresdenSNNDataset,  config):
    print(" === Validation results === ")

    net.train(False)

    conf_truth = []
    conf_pred = []
    epsilon = 0.20

    with torch.no_grad():
        device = config["DEVICE"]
        start_time = time.time()
        models_checked = {}
        cameras = set([(str(x), label) for (x, label) in image_set.image_paths])
        pre_selected = random.sample(cameras, k=9)
        for x, truth in pre_selected:
            name = str(x).split("/")[-2]
            base_image = image_set.images[PosixPath(x)]
            base_image = base_image[None, :]
            same_histo = []
            diff_histo = []
            for (y, label) in image_set.image_paths:
    
                compare_image = image_set.images[PosixPath(y)]
                compare_image = compare_image[None, :]
            
                pred = net(base_image.to(device), compare_image.to(device))
    
                # print(pred.item())
                difference = torch.tanh(pred).item()
                conf_truth.append(truth == label)
                conf_pred.append(difference < epsilon)
                # print(f"{'same' if truth == label else 'diff'} | expect: {int(truth==label)} vs real: {difference:.4f}")
                if truth == label:
                    same_histo.append(difference)
                    # print(f"{x:30} | {y}")
                else:
                    diff_histo.append(difference)
            models_checked[name] = {
                "same" : same_histo,
                "diff" : diff_histo
            }
            print(f"  Validated : {name:<30} | {time.time()-start_time:>5.2f}s")
            start_time = time.time()
                
        models_checked["Summed"] = {
            "same" : [x for (_, v) in models_checked.items() for x in v['same']],
            "diff" : [x for (_, v) in models_checked.items() for x in v['diff']]
        }
     
    multiple_histo(models_checked, config["RESULT_FOLDER"]+f"/histo_multiple_new.svg")
    tp, fp, tn, fn = confusion(torch.Tensor(conf_pred), torch.Tensor(conf_truth))
    print("TN, FP, FN, TP")
    print([tn, fp, fn, tp])
    print(f"Accuracy: {accuracy_score(conf_truth, conf_pred):.4%}")
    print("Confusion matrix: ")
    print(confusion_matrix(conf_truth, conf_pred, normalize='true'))
    print("===== New scores: =====")
    print(f"Prec:   {tp/(tp+fp):.4%}  | TP/(TP+FP)")
    print(f"Recall: {tp/(tp+fn):.4%}  | TP/(TP+FN)")
    

