from ProtINN.segmentation.segmentation import return_superpixels
# from segmentation import return_superpixels
import os
import json
from PIL import Image

def segment_folder_imgs(class_codes, data_folder, out_folder_root, out_folder_child, n_segs = [15,50,80], bg_mode = 'grey'):
    # ensure that out_folder exists
    folder_to_seg = os.path.join(data_folder, out_folder_child)
    if not os.path.isdir(out_folder_root):
        os.mkdir(out_folder_root)
    out_folder = os.path.join(out_folder_root, out_folder_child)
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    # transform format of class information
    with open('/local/work/mpeters/ProtINN/data/imagenet_class_index.json', 'r') as f:
        classes = json.load(f)
    class_dict = {}
    for c in classes:
        class_dict[classes[c][0]] = classes[c][1]

    # for all classes, create a folder, create segments and save those in the respective folders
    for class_code in class_codes:
        print(f"{class_dict[class_code]}: {class_code}")
        class_folder = os.path.join(folder_to_seg, class_code)
        class_out = os.path.join(out_folder, class_code)
        if not os.path.isdir(class_out):
            os.mkdir(class_out)

            filedir = os.listdir(class_folder)
            for i, file in enumerate(filedir):
                img = Image.open(os.path.join(class_folder, file))
                spxs, ptch = return_superpixels(img, n_segs = n_segs, bg_mode = bg_mode)
                maxnr = len(spxs)-1
                for j, s in enumerate(spxs):
                    filename = f"{file.split('.')[0]}_{str(j).rjust(len(str(maxnr)), '0')}.JPEG"
                    s = s.convert('RGB')
                    s.save(os.path.join(class_out, filename))
        else:
            print(f"{class_out} folder already exists. Thus class {class_code} is probably already segmented.")
