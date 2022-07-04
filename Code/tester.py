# Imports
from collections import defaultdict

from SNN import *
from helper import *
from splitter import *



def test_with_loader(net, dataloader, config):
    print(" === Testing results === ")

    net.train(False)
    conf_truth = []
    conf_pred = []
    epsilon = 0.5
    DEVICE = config["DEVICE"]

    models_checked = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for i, (img0, img1, gt, f1, f2) in enumerate(dataloader, 1):  
            # img0, img1 are list of images
            # print(f"{i:>4} / {len(dataloader.dataset)} | {f1[0].split('/')[0]:30} vs {f2[0].split('/')[0]:30}")

            # Send the images and labels to CUDA
            img0, img1, gt = img0.to(DEVICE), img1.to(DEVICE), gt.to(DEVICE)

            # Pass in the images into the network and obtain the outputs
            # The amount of images passed is equal to the batch size
            prediction = net(img0, img1)
            difference = torch.tanh(prediction).item()
            difference = torch.clamp(prediction, max=0.99999).item()
            conf_truth.append(gt)

            # Could be changed into difference >= epsilon, however this expression is a bit more clear
            pred_label = 0. if difference < epsilon else 1
            conf_pred.append(pred_label)
           
            basename = "_".join(str(f1[0]).split("_")[:2])
            if gt:
                # The ground truth says they are different
                models_checked[basename]['diff'].append(difference)
            else:
                models_checked[basename]['same'].append(difference)
    models_checked["Summed"] = {
        "same": [x for (_, v) in models_checked.items() for x in v['same']],
        "diff": [x for (_, v) in models_checked.items() for x in v['diff']]
    }

    multiple_histo(models_checked, epsilon, 
            config["RESULT_FOLDER"]+f"/histo_multiple_{config['UID']}.svg")
    summed_histo(models_checked["Summed"], epsilon,
            config["RESULT_FOLDER"]+f"/histo_summed_{config['UID']}.svg")
    print_scores(torch.Tensor(conf_pred), torch.Tensor(conf_truth))