# Imports
from pathlib import PosixPath
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

import time

from SNN import *
from helper import *
from splitter import *

def test_model(net: SiameseNetwork, dataloader: DataLoader, image_set: DresdenSNNDataset,  config):
    print(" === Testing results === ")
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
            print(f"  Tested : {name:<30} | {time.time()-start_time:>5.2f}s")
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
    print(f"Prec:   {tp/(tp+fp):>2.4%}  | TP/(TP+FP)")
    print(f"Recall: {tp/(tp+fn):>2.4%}  | TP/(TP+FN)")
    
def test_with_loader(net, dataloader, config):
    net.train(False)
    conf_truth = []
    conf_pred = []
    epsilon = 0.45
    DEVICE = config["DEVICE"]

    models_checked = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for i, (img0, img1, gt, f1, f2) in enumerate(dataloader, 1): #img0, img1 are list of images
            # print(f"{i:>4} / {len(dataloader.dataset)} | {f1[0].split('/')[0]:30} vs {f2[0].split('/')[0]:30}")
            
            # Send the images and labels to CUDA
            img0, img1, gt = img0.to(DEVICE), img1.to(DEVICE), gt.to(DEVICE)
            
            # Pass in the two images into the network and obtain two outputs
            prediction = net(img0, img1)
            difference = torch.tanh(prediction).item()
            difference = torch.clamp(prediction, max=1).item()
            conf_truth.append(gt)
            conf_pred.append(difference < epsilon)

            if gt: 
                # The ground truth says they are different
                models_checked[f1[0].split('/')[0]]['diff'].append(difference)
            else:
                models_checked[f1[0].split('/')[0]]['same'].append(difference)
    summing = dict()
    summing['model'] = {"same": [0], 'diff': [1]}
    summing["Summed"] = {
        "same" : [x for (_, v) in models_checked.items() for x in v['same']],
        "diff" : [x for (_, v) in models_checked.items() for x in v['diff']]
    }    
    
    multiple_histo(summing, config["RESULT_FOLDER"]+f"/histo_multiple_new.svg")
    tp, fp, tn, fn = confusion(torch.Tensor(conf_pred), torch.Tensor(conf_truth))
    print("TN, FP, FN, TP")
    print([tn, fp, fn, tp])
    # print(f"Accuracy: {accuracy_score(conf_truth, conf_pred):.4%}")
    # print("Confusion matrix: ")
    # print(confusion_matrix(conf_truth, conf_pred, normalize='true'))
    # print("===== New scores: =====")
    print(f"Prec:   {tp/(tp+fp):>2.4%}  | TP/(TP+FP)")
    print(f"Recall: {tp/(tp+fn):>2.4%}  | TP/(TP+FN)")
                


           
