# IMPORT MODULES
# base modules needed in every file
import os
import bz2
import pickle
import torch
import numpy as np
from datetime import datetime
import argparse
import logging

# trained model
from ibinn_imagenet.model.classifiers.invertible_imagenet_classifier import trustworthy_gc_beta_8

# needed functions
from ProtINN.segmentation.segment_test_img import segment_folder_imgs
from ProtINN.model.activations import get_activations
from ProtINN.data.surrogate_target import create_targets
from ProtINN.classification.similarity_mapping import matrix_sim_mapping
from ProtINN.visualisation.prediction_heatmap import make_heatmap

# needed classes
from ProtINN.data.dataset import ImagenetSegments, ImagenetSurrogateLabels, ImagenetTarget, ImagenetSurrogate
from ProtINN.model.surrogate import PredictionLayer
from ProtINN.data.vars_object import VarsObject


'''How to run:
python evaluating.py config_file_path
'''

# TODO run: python evaluating.py Other/configs/eval_config_20220524.ini
# TODO remove incl function
from memory_profiler import profile
@profile
def func():
    # KEEP TRACK OF RUNTIME
    start_time = datetime.now()
    current_time = start_time

    # LOAD CONFIG FILE FROM ARGUMENT
    parse = argparse.ArgumentParser()
    parse.add_argument("config_file")
    args = parse.parse_args()
    print(f"{args.config_file=}")
    Vars = VarsObject(args.config_file, 'eval')

    logging.basicConfig(filename=f"{Vars.settings_path}/output.txt", level=logging.DEBUG, format='')
    logging.info(f"{args.config_file=}")
    print(f"time for setup: {datetime.now()-current_time}")
    logging.info(f"time for setup: {datetime.now()-current_time}")
    current_time = datetime.now()

    # LOAD DATA
    # segment data if requested (/ needed)
    # segment images in vars.folder_to_segment and save segments to vars.data_folder_root/vars.out_folder_child
    # segment images into vars.n_segs number of images with vars.bg_mode as background mode (grey or blurred)
    if Vars.segment_images:
        segment_folder_imgs(Vars.class_codes, Vars.folder_to_segment, Vars.data_folder_root, 'val', Vars.n_segs, Vars.bg_mode)
        print(f"time for segmenting images: {datetime.now()-current_time}")
        logging.info(f"time for segmenting images: {datetime.now()-current_time}")
        current_time = datetime.now()

    # load segments in vars.data_folder_root/val directory to dataset and dataloader
    val_folder = os.path.join(Vars.data_folder_root, 'val')
    data = ImagenetSegments(val_folder, Vars, Vars.eval_batchsize)
    data_loader = data.data_loader
    print(f"time for loading segment dataset: {datetime.now()-current_time}")
    logging.info(f"time for loading segment dataset: {datetime.now()-current_time}")
    current_time = datetime.now()


    get_bb_output = Vars.surrogate_target in ['weighted', 'multiplied', 'blackbox']
    if get_bb_output:
        # GET FULL IMAGE BLACKBOX PREDICTIONS FOR MODEL TARGET
        pred_pickle = Vars.ibinn_evaloutput_pickle
        if os.path.exists(pred_pickle):
            with bz2.BZ2File(pred_pickle, 'r') as f:
                pred_dict = pickle.load(f)
            print(f"{pred_pickle} loaded")
            logging.info(f"{pred_pickle} loaded")
        else:
            beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
            beta_8_model = beta_8_model.to('cuda:0')
            beta_8_model.eval()

            target_dataset = ImagenetTarget(os.path.join(Vars.folder_to_segment,'val'), Vars, 50)
            target_eval_loader = torch.utils.data.DataLoader(target_dataset, batch_size=target_dataset.batch_size, shuffle=True, num_workers=12, pin_memory=False, sampler=None)

            class_ids = [int(Vars.imagenet_classes[c][0]) for c in Vars.class_codes]
            pred_dict = create_targets(target_eval_loader, beta_8_model, class_ids)

            with bz2.BZ2File(pred_pickle, 'w') as f:
                pickle.dump(pred_dict, f)
            print(f"pred_dict saved to {pred_pickle}")
            logging.info(f"pred_dict saved to {pred_pickle}")

        print(f"time for creating/loading target dataset: {datetime.now()-current_time}")
        logging.info(f"time for creating/loading target dataset: {datetime.now()-current_time}")
        current_time = datetime.now()

    # GET SIMILARITY MAPPING
    if not os.path.exists(Vars.similarity_mapping_pickle_eval):
        # GET ACTIVATIONS OF SEGMENTS
        # if done before, load:
        if os.path.exists(Vars.acts_dict_pickle_eval):
            with bz2.BZ2File(Vars.acts_dict_pickle_eval, 'r') as f:
                acts_dict = pickle.load(f)
            print(f"{Vars.acts_dict_pickle_eval} loaded")
            logging.info(f"{Vars.acts_dict_pickle_eval} loaded")

        # otherwise, get actications and save to dictionary:
        else:
            # load model
            beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
            model = beta_8_model.model

            acts_fc, _, img_paths = get_activations(data_loader, model, Vars.cuda_dev)
            acts_dict = {}
            for img, a_fc in zip(img_paths, acts_fc):
                acts_dict[img] = a_fc

            with bz2.BZ2File(Vars.acts_dict_pickle_eval, 'w') as f:
                pickle.dump(acts_dict, f)
            print(f"acts_dict saved to {Vars.acts_dict_pickle_eval}")
            logging.info(f"acts_dict saved to {Vars.acts_dict_pickle_eval}")

        print(f"time for creating/loading segment activations: {datetime.now()-current_time}")
        logging.info(f"time for creating/loading segment activations: {datetime.now()-current_time}")
        current_time = datetime.now()



    # CREATE / LOAD PROTOTYPES
    if os.path.exists(Vars.kmeans_conceptpickle_full):
        with bz2.BZ2File(Vars.kmeans_conceptpickle_full, 'r') as f:
            kmeans_concept_dict = pickle.load(f)
        print(f"{Vars.kmeans_conceptpickle_full} loaded")
        logging.info(f"{Vars.kmeans_conceptpickle_full} loaded")

        print(f"time for loading concepts: {datetime.now()-current_time}")
        logging.info(f"time for loading concepts: {datetime.now()-current_time}")
        current_time = datetime.now()
    else:
        raise ValueError(f'Could not load prototypes: file {Vars.kmeans_conceptpickle_full} does not exist')


    # get normalized similarity scores
    # if sim-scores not calculated -> calculate -> save
    # else -> load sim-scores
    if os.path.exists(Vars.similarity_mapping_pickle_eval):
        with bz2.BZ2File(Vars.similarity_mapping_pickle_eval, 'r') as f:
            data_dict = pickle.load(f)
        print(f"{Vars.similarity_mapping_pickle_eval} loaded")
        logging.info(f"{Vars.similarity_mapping_pickle_eval} loaded")
    else:
        if os.path.exists(Vars.acts_dict_pickle_small):
            with bz2.BZ2File(Vars.acts_dict_pickle_small, 'r') as f:
                acts_dict_train = pickle.load(f)
            print(f"{Vars.acts_dict_pickle_small} loaded")
            logging.info(f"{Vars.acts_dict_pickle_small} loaded")
        else:
            raise ValueError(f'Could not load training data activations: file {Vars.acts_dict_pickle} does not exist')

        data_dict = matrix_sim_mapping(acts_dict, kmeans_concept_dict, acts_dict_train, Vars)
        
        with bz2.BZ2File(Vars.similarity_mapping_pickle_eval, 'w') as f:
            pickle.dump(data_dict, f)
        print(f"data_dict saved to {Vars.similarity_mapping_pickle_eval}")
        logging.info(f"data_dict saved to {Vars.similarity_mapping_pickle_eval}")

    print(f"time for creating/loading similarity mapping: {datetime.now()-current_time}")
    logging.info(f"time for creating/loading similarity mapping: {datetime.now()-current_time}")
    current_time = datetime.now()


    # evaluate surrogate model
    # 1. load classifier
    # 2. load regression models
    # compare outputs / classifications to blackbox
    if get_bb_output:
        surrogate_dataset_eval = ImagenetSurrogate(data_dict, pred_dict)
    else:
        surrogate_dataset_eval = ImagenetSurrogateLabels(data_dict)

    surrogate_batchsize = Vars.eval_batchsize
    surrogate_loader_eval = torch.utils.data.DataLoader(surrogate_dataset_eval, batch_size=surrogate_batchsize, shuffle=True)

    pred_layer = PredictionLayer(len(kmeans_concept_dict['concepts']), len(Vars.class_codes))
    pred_layer.load_state_dict(torch.load(Vars.surrogate_checkpoint))
    pred_layer.eval()

    # total = [0,0,0,0]
    # top1 = [0,0,0,0]
    # top2 = [0,0,0,0]
    # top3 = [0,0,0,0]
    # top4 = [0,0,0,0]
    repr_acc = 0
    act_acc = 0
    bb_acc = 0
    total = 0

    n = len(Vars.class_codes)
    repr_matrix = np.zeros((n,n))
    acc_matrix = np.zeros((n,n))
    ibinn_matrix = np.zeros((n,n))


    class_ids = {l: i for i, l in enumerate(Vars.class_codes)}


    print(f"time for loading model: {datetime.now()-current_time}")
    logging.info(f"time for loading model: {datetime.now()-current_time}")
    current_time = datetime.now()

    ls = [0,0,0,0,0,0,0,0]
    ls2 = [0,0,0,0,0,0,0,0]
    for i, data in enumerate(surrogate_loader_eval):
        inputs, targets, paths, og_paths = data

        paths_t = tuple([list(x) for x in tuple(zip(*paths))])

        outputs = pred_layer(inputs)

        if get_bb_output:
            targets = targets.to('cpu').detach()

        for t, y, p in zip(targets, outputs, og_paths):
            total += 1

            l = p.split('/')[0]

            if torch.argmax(y).item() == class_ids[l]:
                act_acc += 1
            

            acc_matrix[class_ids[l]][torch.argmax(y).item()] += 1
            
            if get_bb_output:
                if torch.argmax(t).item() == torch.argmax(y).item():
                    repr_acc += 1

                if torch.argmax(t).item() == class_ids[l]:
                    bb_acc += 1
            
                repr_matrix[torch.argmax(t).item()][torch.argmax(y).item()] += 1
                ibinn_matrix[torch.argmax(t).item()][class_ids[l]] += 1

    print(f"{act_acc/total=}")
    print(f"{acc_matrix=}")
    logging.info(f"{act_acc/total=}")
    logging.info(f"{acc_matrix=}")

    if get_bb_output:
        print(f"{repr_acc/total=}")
        print(f"{bb_acc/total=}")

        print(f"{repr_matrix=}")
        print(f"{ibinn_matrix=}")

        logging.info(f"{repr_acc/total=}")
        logging.info(f"{bb_acc/total=}")

        logging.info(f"{repr_matrix=}")
        logging.info(f"{ibinn_matrix=}")

    print(f"time for calculating accuracy: {datetime.now()-current_time}")
    logging.info(f"time for calculating accuracy: {datetime.now()-current_time}")
    current_time = datetime.now()

    make_heatmap(acc_matrix, f'{Vars.target_type}_accuracy_wrt_labels', Vars)
    if get_bb_output:
        make_heatmap(repr_matrix, f'{Vars.target_type}_accuracy_wrt_blackbox', Vars)
        make_heatmap(ibinn_matrix, f'{Vars.target_type}_accuracy_of_blackbox', Vars)

    print(f"time for creating prediction matrices: {datetime.now()-current_time}")
    logging.info(f"time for creating prediction matrices: {datetime.now()-current_time}")
    current_time = datetime.now()


if __name__ == '__main__':
    func()

