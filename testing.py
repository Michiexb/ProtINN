# import modules
import bz2
import pickle
import os
import shutil
import argparse
import json

import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms as T
from torch.autograd import Variable

from ibinn_imagenet.model.classifiers.invertible_imagenet_classifier import trustworthy_gc_beta_8

from ProtINN.data.vars_object import VarsObject
from ProtINN.model.surrogate import PredictionLayer
from ProtINN.data.dataset import SegmentData

from ProtINN.segmentation.segmentation import return_superpixels
from ProtINN.model.activations import get_activations
from ProtINN.classification.similarity_mapping import matrix_sim_mapping
from ProtINN.classification.evaluate import evaluate
from ProtINN.data.surrogate_target import create_targets


# TODO find among all clusters the one with smallest loss for closest segment (prototype vs cluster center)

# run example:
# create new folder (args.save_dir)
# create images.ini file in this new folder to indicate to-be-used images, or give path to ini file as test_config arg
# if evaluation results are desired, create eval_config.ini file in this new folder to indicate folder of eval data, or give path to ini file as test_config arg
# python testing.py Data/run_settings/2022-04-15_14:48:18_20-classes_400_train Data/evaluation_data/eval_labelmodel_1

# download in Git Bash:
# scp -r -J mpeters@login.ikim.uk-essen.de mpeters@g1-6.ikim.uk-essen.de:/local/work/mpeters/Data/evaluation_data/eval_labelmodel_1 "G:\My Drive\Studie\Thesis\Viz\eval_labelmodel_1"

'''How to run:
python testing.py run_dir save_dir [test_config] [eval_config]
    run_dir: of training model. e.g. Data/run_settings/2022-05-23_21:53:52_20-classes_patch_train
    save_dir: path to output directory
    test_config: path to config file containing to-be-used images. e.g. ProtINN/example_ini_files/images.ini
        if images.ini file exists in save_dir, path does not need to be given
    eval_config: path to config file containing path to evaluation data e.g. ProtINN/example_ini_files/eval_config.ini
        only use if evaluation results (eval accuracy + prediction matrix) are desired
        if eval_config.ini file exists in save_dir, path does not need to be given
'''

# TODO run: python testing.py Data/run_settings/2022-05-24_13:11:19_20-classes_patch_train Data/evaluation_data/memory_testing_20220524 ProtINN/example_ini_files/images.ini ProtINN/example_ini_files/eval_config.ini
# TODO remove incl function
from memory_profiler import profile
@profile
def func():
    parse = argparse.ArgumentParser()
    parse.add_argument("run_dir")
    parse.add_argument("save_dir")
    parse.add_argument("test_config", nargs='?')
    parse.add_argument("eval_config", nargs='?')
    args = parse.parse_args()
    print(f"{args.run_dir=}, {args.save_dir=}, {args.test_config=}, {args.eval_config=}")

    img_dir = os.path.join(args.save_dir, 'images')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)


    # load settings
    Vars = VarsObject(os.path.join(args.run_dir, 'config.ini'), 'test')

    if args.test_config is None:
        test_config = os.path.join(args.save_dir, 'images.ini')
    else:
        test_config = args.test_config
    img_paths = Vars.classification_objects(test_config)
        #   img_path -> TODO: turn into list
        # the surrogate model TODO: save in this folder


    # load segments when using files from original imagenet dataset in ImageFolder shape, otherwise always segment the images
    all_seg_paths = []
    all_spxs = []
    for im in img_paths:
        sub_dir = '/'.join(im.split('/')[-3:-1])
        im_id = im.split('/')[-1].split('.')[0]

        # create new folder for segments of given image
        img_subdir = os.path.join(img_dir, im_id)
        if not os.path.exists(img_subdir):
            os.mkdir(img_subdir)

        if not os.path.exists(os.path.join(img_dir, im.split('/')[-1])):
            shutil.copy(im, img_dir)

        seg_dir = os.path.join(Vars.data_folder_root, sub_dir)
        if os.path.exists(seg_dir):
            # find all segment files and copy to vis folder
            seg_files = sorted([x for x in os.listdir(seg_dir) if x[:len(im_id)] == im_id])
            seg_paths = sorted([os.path.join(seg_dir, x) for x in os.listdir(seg_dir) if x[:len(im_id)] == im_id])
        else:
            seg_files = []
            
        if len(seg_files) > 0:
            spxs = []
            for s in seg_paths:
                shutil.copy(s, img_subdir)
                spxs.append(Image.open(s))
        else:
            imgg = Image.open(im)
            spxs, _ = return_superpixels(imgg, n_segs = Vars.n_segs, bg_mode = Vars.bg_mode)
            maxnr = len(spxs)
            seg_paths = []
            for j, s in enumerate(spxs):
                filename = f"{im_id}_{str(j).rjust(len(str(maxnr)), '0')}.JPEG"
                seg_files.append(filename)
                s.save(os.path.join(img_subdir, filename))
                seg_paths.append(os.path.join(img_subdir, filename))
        
        all_seg_paths.extend(seg_paths)
        all_spxs.extend(spxs)

    seg_paths = all_seg_paths
    spxs = all_spxs


    # create custom dataset and loader for segments
    dataset = SegmentData(spxs, seg_paths)
    # batchsize = 10
    data_loader = DataLoader(dataset, Vars.eval_batchsize, shuffle=False, num_workers=12, pin_memory=False, sampler=None)

    # load inn model
    # get activations of segments
    beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
    model = beta_8_model.model

    # get segment latent space
    acts_fc, _, seg_paths = get_activations(data_loader, model, Vars.cuda_dev)
    acts_dict = {}
    for img, a_fc in zip(seg_paths, acts_fc):
        acts_dict[img] = a_fc

    # load prototypes
    concept_pickle = Vars.kmeans_conceptpickle_full
    if os.path.exists(concept_pickle):
        with bz2.BZ2File(concept_pickle, 'r') as f:
            kmeans_concept_dict = pickle.load(f)
        print(f"{concept_pickle} loaded")
    else:
        raise ValueError(f'Could not load concept dict: file {concept_pickle} does not exist')


    # copy closest seg of each concept to concepts folder
    concept_folder = os.path.join(args.save_dir, 'concepts')
    if not os.path.exists(concept_folder):
        os.mkdir(concept_folder)

    concept_dict = {}
    for c in kmeans_concept_dict['concepts']:
        medoid = kmeans_concept_dict[c]['image_paths'][0]
        medoid_fc = os.path.join(concept_folder, medoid.split('/')[-1])
        if not os.path.exists(medoid_fc):
            shutil.copy(medoid, concept_folder)
        concept_dict[c] = medoid_fc
    fp = os.path.join(args.save_dir, f"concept_files.txt")
    with open(fp, 'w') as f:
        f.write(json.dumps(concept_dict))


    # copy class dict to dir
    if not os.path.exists(os.path.join(args.save_dir, Vars.class_index_file.split('/')[-1])):
        shutil.copy(Vars.class_index_file, args.save_dir)


    # get normalized similarity scores
    acts_dict_train_pickle = Vars.acts_dict_pickle_small # add _small (after creating that pickle for 400 c)
    if os.path.exists(acts_dict_train_pickle):
        with bz2.BZ2File(acts_dict_train_pickle, 'r') as f:
            acts_dict_train = pickle.load(f)
        print(f"{acts_dict_train_pickle} loaded")
    else:
        raise ValueError(f'Could not load training data activations: file {acts_dict_train_pickle} does not exist')

    data_dict = matrix_sim_mapping(acts_dict, kmeans_concept_dict, acts_dict_train, Vars)


    sim_dir = os.path.join(args.save_dir, 'sim_dicts')
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)
    for im_id, img_path in enumerate(data_dict['image_paths']):
        imsim = list(data_dict['similarity'][im_id])
        seg_list = list(data_dict['segment_paths'][im_id])
        
        concept_sim_dict = {}
        for concept, segm, simsc in zip(kmeans_concept_dict['concepts'], seg_list, imsim):
            concept_sim_dict[concept] = (segm, simsc.item())

        # save to file
        img_name = img_path.split('/')[-1].split('.')[0]
        fp = os.path.join(sim_dir, f"{img_name}_sim_dict.txt")
        with open(fp, 'w') as f:
            f.write(json.dumps(concept_sim_dict))


    # load surrogate model
    pred_layer = PredictionLayer(Vars.n_clusters_kmeans, len(Vars.class_codes))

    checkpointfiles = [x for x in os.listdir(args.run_dir) if x.endswith('.pt')]
    if len(checkpointfiles) > 0:
        surrogate_checkpoint = os.path.join(args.run_dir, checkpointfiles[0])
    else:
        surrogate_checkpoint = Vars.surrogate_checkpoint # TODO: replace with save file for new runs (or check for checkpoint file in run_dir and load Vars path if does not exist)
    pred_layer.load_state_dict(torch.load(surrogate_checkpoint))
    pred_layer.eval()


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


    # save to file
    fp = os.path.join(args.save_dir, f"weights.txt")
    with open(fp, 'w') as f:
        f.write(json.dumps(wdict))


    # surrogate model input
    class_score_dir = os.path.join(args.save_dir, 'class_scores')
    if not os.path.exists(class_score_dir):
        os.mkdir(class_score_dir)

    xlist = Variable(torch.FloatTensor(data_dict['similarity']))
    for x, img_path in zip(xlist, data_dict['image_paths']):
        x = (x * 10**2).round() / (10**2)
        output = pred_layer(x)


        output_dict = {cl: y for cl, y in zip(Vars.class_codes, output.tolist())}
        # save to file
        img_id = img_path.split('/')[-1].split('.')[0]
        fp = os.path.join(class_score_dir, f"{img_id}_class_scores.txt")
        with open(fp, 'w') as f:
            f.write(json.dumps(output_dict))


    eval_ini_path = os.path.join(args.save_dir, 'eval_config.ini')
    if os.path.exists(eval_ini_path) or args.eval_config is not None:
        if args.eval_config is None:
            acc, matrix = evaluate(Vars, pred_layer, eval_ini_path)
        else:
            acc, matrix = evaluate(Vars, pred_layer, args.eval_config)
        print(type(matrix))
        pred_matrix = matrix.tolist()

        # save acc and matrix to files
        eval_dir = os.path.join(args.save_dir, 'eval_results')
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        eval_dict = {}
        eval_dict['accuracy'] = acc
        eval_dict['pred_matrix'] = pred_matrix

        eval_dict_file = os.path.join(eval_dir, 'model_accuracy.txt')
        with open(eval_dict_file, 'w') as f:
            f.write(json.dumps(eval_dict))



    # output:
        # dir with name vars.class_list_file.split('.')[0]
            # dir: concepts
                # files: closest img per concept # TODO mapping concept id to file
            # dir: images
                # files: {img_id}.JPEG
                # dirs: {img_id}
                    # files: segment_{i}.JPEG
            # dir: sim_dicts
                # files: {img_id}_max_sim.txt
            # file: weights.txt
            # file: class_scores.txt
            # file: class_dict.txt # TODO


    # for each image, also get predictions of black box
    beta_8_model = beta_8_model.to('cuda:0')
    beta_8_model.eval()

    class TargetData():
        def __init__(self, path_list):
            self.data = [Image.open(p) for p in path_list]
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
            img = self.data[index].convert('RGB')
            X = self.transform(img)
            P = self.paths[index]
            return X,P

    target_dataset = TargetData(img_paths)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=Vars.eval_batchsize, shuffle=True, num_workers=12, pin_memory=False, sampler=None)
    
    class_ids = [int(Vars.imagenet_classes[c][0]) for c in Vars.class_codes]
    pred_dict = create_targets(target_loader, beta_8_model, class_ids)
    target_dict = {k: v.tolist() for k,v in pred_dict.items()}
    print(target_dict)

    fp2 = os.path.join(args.save_dir, f"target_dict.txt")
    with open(fp2, 'w') as f:
        f.write(json.dumps(target_dict))


if __name__ == '__main__':
    func()