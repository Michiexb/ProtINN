
import bz2
import pickle
import argparse
import os
import random
from datetime import datetime
import sys

from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from ibinn_imagenet.model.classifiers.invertible_imagenet_classifier import trustworthy_gc_beta_8

from ProtINN.data.vars_object import VarsObject
from ProtINN.model.surrogate import PredictionLayer
from ProtINN.data.dataset import SegmentData

from ProtINN.segmentation.segmentation import return_superpixels
from ProtINN.model.activations import get_activations
from ProtINN.classification.similarity_mapping import matrix_sim_mapping
from ProtINN.evaluation.get_test_segs import test_segs, model_properties, add_blur, add_prototype

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

"""How to run:
python correctness.py run_dir save_dir [test_config]
    run_dir: of training model. e.g. Data/run_settings/2022-05-23_21:53:52_20-classes_patch_train
    save_dir: path to output directory
    test_config: path to config file containing to-be-used images. e.g. ProtINN/example_ini_files/images.ini
        if images.ini file exists in save_dir, path does not need to be given
"""

def correctness_func():
    # KEEP TRACK OF RUNTIME
    start_time = datetime.now()
    current_time = start_time

    parse = argparse.ArgumentParser()
    parse.add_argument("run_dir") # which contains config.ini
    parse.add_argument("save_dir") # to save result images to
    parse.add_argument("test_config", nargs='?') # images to adjust, either given as argument or created as test_config.ini in save_dir
    args = parse.parse_args()
    print(f"{args.run_dir=}, {args.save_dir=}, {args.test_config=}")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # load settings
    Vars = VarsObject(os.path.join(args.run_dir, 'config.ini'), 'correctness')

    # read test_config
    if args.test_config is None:
        test_config = os.path.join(args.save_dir, 'correctness_images.ini')
    else:
        test_config = args.test_config
    img_paths = Vars.classification_objects(test_config)

    print(f"time for setup: {datetime.now()-current_time}")
    current_time = datetime.now()

    correctness_pickle = os.path.join(args.save_dir, 'correctness_data.pickle.compressed')
    if os.path.exists(correctness_pickle):
        with bz2.BZ2File(correctness_pickle, 'r') as f:
            spxs, seg_paths, segloc_dict = pickle.load(f)
        print(f"{correctness_pickle} loaded")
    else:
        # get dataset of test image segments, including the segment locations
        spxs, seg_paths, segloc_dict = test_segs(Vars, img_paths)
        with bz2.BZ2File(correctness_pickle, 'w') as f:
            pickle.dump([spxs, seg_paths, segloc_dict], f)
        print(f"[spxs, seg_paths, seg_dict] saved to {correctness_pickle}")

        
    dataset = SegmentData(spxs, seg_paths)
    seg_batchsize = 10
    data_loader = DataLoader(dataset, seg_batchsize, shuffle=False, num_workers=12, pin_memory=False, sampler=None)
            




    print(f"time for getting segments: {datetime.now()-current_time}")
    current_time = datetime.now()

    # load inn model
    # get activations of segments
    beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
    model = beta_8_model.model

    print(f"time for loading model: {datetime.now()-current_time}")
    current_time = datetime.now()

    # get segment latent space
    acts_fc, _, image_paths = get_activations(data_loader, model, torch.device(f"cuda:{Vars.cuda_nr}"))
    acts_dict = {}
    for img, a_fc in zip(image_paths, acts_fc):
        acts_dict[img] = a_fc

    print(f"time for getting activations: {datetime.now()-current_time}")
    current_time = datetime.now()

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

    print(f"time for loading and making concept dict: {datetime.now()-current_time}")
    current_time = datetime.now()

    # get normalized similarity scores
    acts_dict_train_pickle = Vars.acts_dict_pickle_small # add _small (after creating that pickle for 400 c)
    if os.path.exists(acts_dict_train_pickle):
        with bz2.BZ2File(acts_dict_train_pickle, 'r') as f:
            acts_dict_train = pickle.load(f)
        print(f"{acts_dict_train_pickle} loaded")
    else:
        raise ValueError(f'Could not load training data activations: file {acts_dict_train_pickle} does not exist')

    print(f"time for loading training activations: {datetime.now()-current_time}")
    current_time = datetime.now()

    data_dict = matrix_sim_mapping(acts_dict, kmeans_concept_dict, acts_dict_train, Vars)

    print(f"time for getting similarity scores: {datetime.now()-current_time}")
    current_time = datetime.now()


    # load surrogate model
    pred_layer = PredictionLayer(Vars.n_clusters_kmeans, len(Vars.class_codes))

    surrogate_checkpoint = Vars.surrogate_checkpoint # replace with save file for new runs (or check for checkpoint file in run_dir and load Vars path if does not exist)
    pred_layer.load_state_dict(torch.load(surrogate_checkpoint))
    pred_layer.eval()

    print(f"time for loading model: {datetime.now()-current_time}")
    current_time = datetime.now()

    # get model weights and image class prediction scores
    w_dict, output_dict = model_properties(Vars, pred_layer, data_dict)

    print(f"time for getting model properties and output: {datetime.now()-current_time}")
    current_time = datetime.now()


    changed_dict = {}
    # full path needed:
    for i, img_path in enumerate(data_dict['image_paths']):
        img_path = os.path.join(Vars.folder_to_segment, '/'.join(img_path.split('/')[-3:]))

        imgg = Image.open(img_path)

        # save original image to folder for reference
        img_name = img_path.split('/')[-1].split('.')[0]
        og_img_path = os.path.join(args.save_dir, f'{img_name}_original_image.JPEG')
        imgg.save(og_img_path)


        img_subpath = '/'.join(img_path.split('/')[-3:])
        predictions = output_dict[img_subpath]
        cls1 = max(predictions, key=predictions.get)

        blurred_img = add_blur(w_dict, cls1, data_dict, segloc_dict, imgg, i)

        # save changed_img to file
        blurred_img = blurred_img.convert('RGB')
        blurred_img_path = os.path.join(args.save_dir, f'{img_name}_blurred_{cls1}.JPEG')
        blurred_img.save(blurred_img_path)

        prediction_tuples = [(k,v) for k,v in predictions.items()]
        prediction_tuples.sort(key=lambda x: x[1])
        n_lowest_classes = 5
        lowest_classes = prediction_tuples[:n_lowest_classes]
        rand_l_id = random.randint(0,n_lowest_classes-1) #random.randint max is included, np.random.randint max is excluded
        cls2 = lowest_classes[rand_l_id][0]

        backgroundimg = add_prototype(w_dict, cls2, kmeans_concept_dict, imgg)

        # save changed_img to file
        additioned_img = backgroundimg.convert('RGB')
        additioned_img_path = os.path.join(args.save_dir, f'{img_name}_added_{cls2}.JPEG')
        additioned_img.save(additioned_img_path)

        changed_dict[img_path] = {'original': (og_img_path, imgg), 'blurred': (blurred_img_path, blurred_img), 'additioned': (additioned_img_path, additioned_img), 'classes': (cls1, cls2)}
        

    print(f"time for blurring and augmenting all images: {datetime.now()-current_time}")
    current_time = datetime.now()


    # add new predictions
    change_count_blur = 0
    change_count_add = 0
    changed_add_n = []
    for item in changed_dict:
        item_dict = changed_dict[item]


        # prediction of original
        img_subpath = '/'.join(item.split('/')[-3:])
        predictions = output_dict[img_subpath]


        # prediction of blurred
        spxs_blur, _ = return_superpixels(item_dict['blurred'][1], n_segs = Vars.n_segs, bg_mode = Vars.bg_mode)
        seg_paths_blur = [f'seg_{i}' for i in range(0, len(spxs_blur), 1)]
        dataset_blur = SegmentData(spxs_blur, seg_paths_blur)
        data_loader_blur = DataLoader(dataset_blur, Vars.eval_batchsize, shuffle=False, num_workers=12, pin_memory=False, sampler=None)

        acts_fc_blur, _, seg_paths_blur = get_activations(data_loader_blur, model, Vars.cuda_dev)
        acts_dict_blur = {}
        for img, a_fc in zip(seg_paths_blur, acts_fc_blur):
            acts_dict_blur[img] = a_fc

        data_dict_blur = matrix_sim_mapping(acts_dict_blur, kmeans_concept_dict, acts_dict_train, Vars)
        xlist_blur = Variable(torch.FloatTensor(data_dict_blur['similarity']))
        x_blur = (xlist_blur[0] * 10**2).round() / (10**2)
        output_blur = pred_layer(x_blur)
        output_dict_blur = {cl: y for cl, y in zip(Vars.class_codes, output_blur.tolist())}

        changed_blur = (max(predictions, key=predictions.get) != max(output_dict_blur, key=output_dict_blur.get))
        if changed_blur:
            change_count_blur += 1


        # prediction of additioned
        spxs_add, _ = return_superpixels(item_dict['additioned'][1], n_segs = Vars.n_segs, bg_mode = Vars.bg_mode)
        seg_paths_add = [f'seg_{i}' for i in range(0, len(spxs_add), 1)]
        dataset_add = SegmentData(spxs_add, seg_paths_add)
        data_loader_add = DataLoader(dataset_add, Vars.eval_batchsize, shuffle=False, num_workers=12, pin_memory=False, sampler=None)

        acts_fc_add, _, seg_paths_add = get_activations(data_loader_add, model, Vars.cuda_dev)
        acts_dict_add = {}
        for img, a_fc in zip(seg_paths_add, acts_fc_add):
            acts_dict_add[img] = a_fc

        data_dict_add = matrix_sim_mapping(acts_dict_add, kmeans_concept_dict, acts_dict_train, Vars)
        xlist_add = Variable(torch.FloatTensor(data_dict_add['similarity']))
        x_add = (xlist_add[0] * 10**2).round() / (10**2)
        output_add = pred_layer(x_add)
        output_dict_add = {cl: y for cl, y in zip(Vars.class_codes, output_add.tolist())}

        predictions_tuplist = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        predictions_classes = [x[0] for x in predictions_tuplist]
        predictions_cls2_id = predictions_classes.index(cls2)
        output_add_tuplist = sorted(output_dict_add.items(), key=lambda x: x[1], reverse=True)
        output_add_classes = [x[0] for x in output_add_tuplist]
        output_add_cls2_id = output_add_classes.index(cls2)
        changed_add = (output_add_cls2_id < predictions_cls2_id)
        if changed_add:
            change_count_add += 1
        n_changed_add = output_add_cls2_id - predictions_cls2_id
        changed_add_n.append(n_changed_add)
        
    print(f"{change_count_blur=}")
    print(f"{change_count_add=}")
    avg_changed_add_n = sum(changed_add_n)/len(changed_add_n)
    print(f"{avg_changed_add_n}")

    print(f"time for getting new predictions: {datetime.now()-current_time}")

    print(f"time for entire run: {datetime.now()-start_time}")

if __name__ == '__main__':
    correctness_func()