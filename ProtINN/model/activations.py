import os
import torch
torch.cuda.empty_cache()

def get_activations(loader, model, cuda_dev):
    count = len(loader)

    model = model.to(cuda_dev)
    model.eval()

    acts = []
    acts_conv = []
    img_paths = []

    for i, (img, lbl, path) in enumerate(loader):
        print(f"batch {i}/{count}")

        img = img.to(cuda_dev)

        with torch.no_grad():
            out_fc, out_conv = model(img)
        out_fc = out_fc.to('cpu').detach()
        out_conv = out_conv.to('cpu').detach()

        acts.extend(out_fc.tolist())
        # acts_conv.extend(out_conv.tolist()) # not needed for clustering, save intermediate to file ?
        img_paths.extend(list(path))

        if i % 20 == 0: # print memory usage
            total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))

    return acts, acts_conv, img_paths