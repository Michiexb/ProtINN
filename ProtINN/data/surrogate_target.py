import torch
import numpy as np

def create_targets(train_loader, model, class_id_list):
    class_ids = np.array(class_id_list)

    pred_dict = {}
    for i, (x, p) in enumerate(train_loader):
        with torch.no_grad():
            x = x.to('cuda:0')
            pred = model(x)
            dist = pred['cluster_distances']
            sim = (torch.tanh((-torch.sqrt(dist)+28)/2)+1)/2

        for j in range(len(p)):
            tanhed_temp = sim[j][class_ids]
            pth = p[j]
            file_pth = pth.split('/')[-1]
            pred_dict[file_pth] = tanhed_temp

    return pred_dict

