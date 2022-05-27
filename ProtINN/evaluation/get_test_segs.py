import os
import numpy as np
import random

from PIL import Image, ImageFilter

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms as T

from ProtINN.segmentation.segmentation import return_superpixels

# TODO: remove option for checking existing segments, since locations are still needed (at least for blur option)
def test_segs(Vars, img_paths):
    # get img segments
    # check if they exist in Vars.data_folder_root
    imgfolder_paths = [os.path.join(Vars.data_folder_root, '/'.join(img.split('/')[-3:]).split('.')[0]) for img in img_paths] #e.g: 'Data/segments' + 'val/n01693334/ILSVRC2012_val_00003722'
    img_has_segs = {og_imgpath: seg_imgpath for og_imgpath, seg_imgpath in zip(img_paths, imgfolder_paths)}
    for og_imgpath in img_has_segs:
        seg_imgpath = img_has_segs[og_imgpath]

        imgdir = '/'.join(seg_imgpath.split('/')[:-1])
        imgname = seg_imgpath.split('/')[-1]

        has_segs = False
        for file in os.listdir(imgdir):
            if file.startswith(imgname):
                has_segs = True
        if not has_segs:
            img_has_segs[og_imgpath] = False
        

    # for img in img_has_segs
        # if False: create
        # else: load based on value path
    all_seg_paths = []
    all_spxs = []
    seg_locs = []

    for im in img_has_segs:
        if True:#not img_has_segs[im]:
            im_id = im.split('/')[-1].split('.')[0]

            imgg = Image.open(im).convert("RGB")
            spxs, patch = return_superpixels(imgg, n_segs = Vars.n_segs, bg_mode = Vars.bg_mode)
            maxnr = len(spxs)
            
            seg_files = []
            for j, s in enumerate(spxs):
                filename = f"{im_id}_{str(j).rjust(len(str(maxnr)), '0')}.JPEG"
                seg_files.append(os.path.join('/'.join(im.split('/')[-3:-1]), filename))

                # check if folder exists and create?
                seg_dir = os.path.join(Vars.data_folder_root, '/'.join(im.split('/')[-3:-1]))
                if not os.path.exists(seg_dir):
                    subdir = '/'.join(seg_dir.split('/')[:-1])
                    if os.path.exists(subdir):
                        os.mkdir(seg_dir)
                    else:
                        seg_dir = os.join(Vars.data_folder_root, 'other')
                        os.mkdir(seg_dir)
                s.save(os.path.join(seg_dir, filename))
            
            all_seg_paths.extend(seg_files)
            all_spxs.extend(spxs)
            seg_locs.extend(patch)

        else:
            # load existing segments, but do we then also have the locations???
            pass

    seg_paths = all_seg_paths
    spxs = all_spxs

    segloc_dict = {p: l for p,l in zip(seg_paths, seg_locs)}

    return spxs, seg_paths, segloc_dict




# ProtINN
# create custom dataset and loader for segments
class SegmentData():
    def __init__(self, img_list, path_list):
        self.data = img_list
        self.paths = path_list
        
        self.img_crop_size = (224, 224)

        self._mu_img = [0.485, 0.456, 0.406]
        self._std_img = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.img_crop_size),
            T.ToTensor(),
            T.Normalize(self._mu_img, self._std_img),
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        X = self.transform(img)
        P = self.paths[index]
        return X,0,P



def model_properties(Vars, pred_layer, data_dict):
    if len(list(pred_layer.parameters())) > 1:
        weights, bias =  pred_layer.parameters()

        wdict = {}
        for cl, w, b in zip(Vars.class_codes, weights, bias):
            wdict[cl] = {f"concept{i+1}": w_item for i, w_item in enumerate(w.tolist())}
            wdict[cl]['bias'] = b.item()
    else:
        weights = list(pred_layer.parameters())[0]

        wdict = {}
        for cl, w in zip(Vars.class_codes, weights):
            wdict[cl] = {f"concept{i+1}": w_item for i, w_item in enumerate(w.tolist())}


    xlist = Variable(torch.FloatTensor(data_dict['similarity']))

    output_dict = {}
    for x, img_path in zip(xlist, data_dict['image_paths']):
        x = (x * 10**2).round() / (10**2)
        output = pred_layer(x)

        partial_output_dict = {cl: y for cl, y in zip(Vars.class_codes, output.tolist())}
        img_subpath = '/'.join(img_path.split('/')[-3:])
        output_dict[img_subpath] = partial_output_dict

    return wdict, output_dict



def add_blur(w_dict, cls1, data_dict, segloc_dict, imgg, nth_img):
    # for this class, calculate all w*sim
    ws = list(w_dict[cls1].values())
    sims = [(x.item(), y) for x,y in zip(data_dict['similarity'][nth_img], data_dict['segment_paths'][nth_img])] # 0 now that it's only done for a single image
    wxsim = [(w*s[0], s[1], w, s[0]) for w,s in zip(ws, sims)] # also save the segment somehow
    wxsim.sort(key=lambda y: y[0], reverse=True)

    # translate segment path to image location (box)
    seg_to_change = wxsim[0][1]
    seg_loc = segloc_dict[seg_to_change]

    # image part occlusion - blurred
    base_img = imgg
    imgw,imgh = imgg.size
    mask = np.zeros((imgh,imgw))
    mask[seg_loc[1]:seg_loc[3],seg_loc[0]:seg_loc[2]] = 1
    mask_img = Image.fromarray(np.uint8(255*mask))
    blurred_part = imgg.filter(ImageFilter.GaussianBlur(5))
    blurred_img = Image.composite(blurred_part, base_img, mask_img)

    return blurred_img



def add_prototype(wdict, cls2, kmeans_concept_dict, imgg):
    # image part addition
    ws = [(k, v) for k,v in wdict[cls2].items()]
    ws.sort(key=lambda y: y[1], reverse=True)
    best_concept = ws[0][0]

    nth_concept_img = random.randint(0,5)
    concept_path = kmeans_concept_dict[best_concept]['image_paths'][nth_concept_img]
    concept_img = Image.open(concept_path)

    imgw,imgh = imgg.size
    neww, newh = (int(0.2*imgw), int(0.2*imgh))
    resized_concept = concept_img.resize((neww,newh))
    wmax, hmax = (imgw-neww, imgh-newh)
    wloc = random.randint(0,wmax)
    hloc = random.randint(0,hmax)

    # paste prototype onto original image at random location
    backgroundimg = imgg
    backgroundimg.paste(resized_concept, (wloc,hloc))

    return backgroundimg


