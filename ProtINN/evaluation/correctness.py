
import bz2
import pickle
import argparse
import os

from PIL import Image

import torch
from torch.autograd import Variable

from ibinn_imagenet.model.classifiers.invertible_imagenet_classifier import trustworthy_gc_beta_8

from ProtINN.data.vars_object import VarsObject
from ProtINN.model.surrogate import PredictionLayer

from ProtINN.model.activations import get_activations
from ProtINN.classification.similarity_mapping import matrix_sim_mapping
from ProtINN.evaluation.get_test_segs import test_segs, model_properties

"""
give some images to adjust
get class prediction scores
find out highest class prediction per image
blur: find out highest contributing segment location and blur
addition: take some low scoring class and find its most important prototype
    for that prototype, grab some segment in cluster close to center
    add that segment to a random location in the original image
for the new images, get new class prediction scores

(make quantative?
blur: how often not highest prediction anymore
addition: how often initial highest not highest anymore, how often added class now highest, how many classes passed on average?)

also get outputs for visualisations as done in testing.py
"""


parse = argparse.ArgumentParser()
parse.add_argument("run_dir") # which contains config.ini
parse.add_argument("save_dir") # to save result images to
parse.add_argument("test_config", nargs='?') # images to adjust, either given as argument or created as test_config.ini in save_dir
args = parse.parse_args()
print(f"{args.run_dir=}, {args.save_dir=}, {args.test_config=}")

# load settings
Vars = VarsObject(os.path.join(args.run_dir, 'config.ini'), 'correctness')

# read test_config
if args.test_config is None:
    test_config = os.path.join(args.save_dir, 'correctness_images.ini')
else:
    test_config = args.test_config
img_paths = Vars.classification_objects(test_config)


# get dataset of test image segments, including the segment locations
data_loader, segloc_dict = test_segs(Vars, img_paths)

# load inn model
# get activations of segments
beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
model = beta_8_model.model

# get segment latent space
acts_fc, _, img_paths = get_activations(data_loader, model, torch.device(f"cuda:{Vars.cuda_nr}"))
acts_dict = {}
for img, a_fc in zip(img_paths, acts_fc):
    acts_dict[img] = a_fc

# load prototypes
concept_pickle = Vars.kmeans_conceptpickle_full
if os.path.exists(concept_pickle):
    with bz2.BZ2File(concept_pickle, 'r') as f:
        kmeans_concept_dict = pickle.load(f)
    print(f"{concept_pickle} loaded")
else:
    raise ValueError(f'Could not load concept dict: file {concept_pickle} does not exist')

concept_dict = {}
for c in kmeans_concept_dict['concepts']:
    medoid = kmeans_concept_dict[c]['image_paths'][0]
    concept_dict[c] = medoid

# get normalized similarity scores
acts_dict_train_pickle = Vars.acts_dict_pickle_small # add _small (after creating that pickle for 400 c)
if os.path.exists(acts_dict_train_pickle):
    with bz2.BZ2File(acts_dict_train_pickle, 'r') as f:
        acts_dict_train = pickle.load(f)
    print(f"{acts_dict_train_pickle} loaded")
else:
    raise ValueError(f'Could not load training data activations: file {acts_dict_train_pickle} does not exist')

data_dict = matrix_sim_mapping(acts_dict, kmeans_concept_dict, acts_dict_train)

print(f"{data_dict['image_paths']=}")
print(f"{img_paths=}")



# load surrogate model
pred_layer = PredictionLayer(Vars.n_clusters_kmeans, len(Vars.class_codes))

surrogate_checkpoint = Vars.surrogate_checkpoint # replace with save file for new runs (or check for checkpoint file in run_dir and load Vars path if does not exist)
pred_layer.load_state_dict(torch.load(surrogate_checkpoint))
pred_layer.eval()


# get model weights and image class prediction scores
# output_dict[img_path] = {cl: y for cl, y in zip(Vars.class_codes, output.tolist())}
w_dict, output_dict = model_properties(Vars, pred_layer, data_dict)

# output dict:  class predictions
# w dict:       weights concepts x classes
# data dict:    sim scores



for img_path in img_paths:
    img = Image.open(img_path)
    # save original image to folder for reference
    img.save(os.path.join(args.save_dir, 'original_image.JPEG'))


# for each image, let cls1 be the class with highest prediction score




# for this class, calculate all w*sim
ws = list(wdict[cls1].values())
sims = [(x.item(), y) for x,y in zip(data_dict['similarity'][0], data_dict['segment_paths'][0])] # 0 now that it's only done for a single image
wxsim = [(w*s[0], s[1]) for w,s in zip(ws, sims)] # also save the segment somehow
wxsim.sort(key=lambda y: y[0], reverse=True)




# translate segment path to image location (box)
seg_to_change = wxsim[0][1]
seg_loc = segloc_dict[seg_to_change]




# TODO to function occlusion
# image part occlusion - blurred
base_img = imgg
imgw,imgh = imgg.size
mask = np.zeros((imgh,imgw))
mask[seg_loc[1]:seg_loc[3],seg_loc[0]:seg_loc[2]] = 1
mask_img = Image.fromarray(np.uint8(255*mask))
blurred_part = imgg.filter(ImageFilter.GaussianBlur(5))
blurred_img = Image.composite(blurred_part, base_img, mask_img)

# save changed_img to file
blurred_img = blurred_img.convert('RGB')
blurred_img.save(os.path.join(eval_folder_path, 'blurred_occlusion.JPEG'))






# TODO to function addition
# image part addition
ws = [(k, v) for k,v in wdict[cls2].items()]
ws.sort(key=lambda y: y[1], reverse=True)
best_concept = ws[0][0]
concept_path = kmeans_concept_dict[best_concept]['image_paths'][0]
concept_img = Image.open(concept_path)
neww, newh = (int(0.2*imgw), int(0.2*imgh))
resized_concept = concept_img.resize((neww,newh))
wmax, hmax = (imgw-neww, imgh-newh)
wloc = random.randint(0,wmax)
hloc = random.randint(0,hmax)

# paste prototype onto original image at random location
backgroundimg = imgg
backgroundimg.paste(resized_concept, (wloc,hloc))

# save changed_img to file
additioned_img = backgroundimg.convert('RGB')
additioned_img.save(os.path.join(eval_folder_path, 'pasted_addition.JPEG'))

