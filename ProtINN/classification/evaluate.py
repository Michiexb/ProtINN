# IMPORT MODULES
# base modules needed in every file
import os
import bz2
import pickle
import torch
import numpy as np

# trained model
from ibinn_imagenet.model.classifiers.invertible_imagenet_classifier import trustworthy_gc_beta_8

# needed functions
from ProtINN.segmentation.segment_test_img import segment_folder_imgs
from ProtINN.model.activations import get_activations
from ProtINN.data.surrogate_target import create_targets
from ProtINN.classification.similarity_mapping import matrix_sim_mapping

# needed classes
from ProtINN.data.dataset import ImagenetSegments, ImagenetTarget, ImagenetSurrogate, ImagenetSurrogateLabels

def evaluate(Vars, pred_model, eval_config):
    # print(f"{pred_model=}")

    # LOAD DATA
    # segment data if requested (/ needed)
    # segment images in vars.folder_to_segment and save segments to vars.data_folder_root/vars.out_folder_child
    # segment images into vars.n_segs number of images with vars.bg_mode as background mode (grey or blurred)
    eval_vars = Vars.evaluation_object(eval_config)

    if eval_vars.segment_images:
        segment_folder_imgs(Vars.class_codes, eval_vars.folder_to_segment, eval_vars.data_folder_root, 'val', Vars.n_segs, Vars.bg_mode)


    # load segments in vars.data_folder_root/val directory to dataset and dataloader
    val_folder = os.path.join(eval_vars.data_folder_root, 'val')
    data = ImagenetSegments(val_folder, Vars, eval_vars.eval_batchsize)
    data_loader = data.data_loader

    
    get_bb_output = Vars.surrogate_target in ['weighted', 'multiplied', 'blackbox']
    if get_bb_output:
        # GET FULL IMAGE BLACKBOX PREDICTIONS FOR MODEL TARGET
        pred_pickle = Vars.ibinn_evaloutput_pickle
        if os.path.exists(pred_pickle):
            with bz2.BZ2File(pred_pickle, 'r') as f:
                pred_dict = pickle.load(f)
            print(f"{pred_pickle} loaded")
        else:
            beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
            beta_8_model = beta_8_model.to('cuda:0')
            beta_8_model.eval()

            target_dataset = ImagenetTarget(os.path.join(eval_vars.folder_to_segment,'val'), Vars, 50)
            target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=target_dataset.batch_size, shuffle=True, num_workers=12, pin_memory=False, sampler=None)

            class_ids = [int(Vars.imagenet_classes[c][0]) for c in Vars.class_codes]
            pred_dict = create_targets(target_loader, beta_8_model, class_ids)

            with bz2.BZ2File(pred_pickle, 'w') as f:
                pickle.dump(pred_dict, f)
            print(f"pred_dict saved to {pred_pickle}")

    # GET SIMILARITY MAPPING
    if not os.path.exists(Vars.similarity_mapping_pickle_eval):
        # GET ACTIVATIONS OF SEGMENTS
        # if done before, load:
        if os.path.exists(Vars.acts_dict_pickle_eval):
            with bz2.BZ2File(Vars.acts_dict_pickle_eval, 'r') as f:
                acts_dict = pickle.load(f)
            print(f"{Vars.acts_dict_pickle_eval} loaded")

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


    # load prototypes
    if os.path.exists(Vars.kmeans_conceptpickle_full):
        with bz2.BZ2File(Vars.kmeans_conceptpickle_full, 'r') as f:
            kmeans_concept_dict = pickle.load(f)
        print(f"{Vars.kmeans_conceptpickle_full} loaded")
    else:
        raise ValueError(f'Could not load prototypes: file {Vars.kmeans_conceptpickle_full} does not exist')


    # get normalized similarity scores
    # if sim-scores not calculated -> calculate -> save
    # else -> load sim-scores
    if os.path.exists(Vars.similarity_mapping_pickle_eval):
        with bz2.BZ2File(Vars.similarity_mapping_pickle_eval, 'r') as f:
            data_dict = pickle.load(f)
        print(f"{Vars.similarity_mapping_pickle_eval} loaded")
    else:
        if os.path.exists(Vars.acts_dict_pickle_small):
            with bz2.BZ2File(Vars.acts_dict_pickle_small, 'r') as f:
                acts_dict_train = pickle.load(f)
            print(f"{Vars.acts_dict_pickle_small} loaded")
        else:
            raise ValueError(f'Could not load training data activations: file {Vars.acts_dict_pickle} does not exist')

        data_dict = matrix_sim_mapping(acts_dict, kmeans_concept_dict, acts_dict_train, Vars)
        
        with bz2.BZ2File(Vars.similarity_mapping_pickle_eval, 'w') as f:
            pickle.dump(data_dict, f)
        print(f"data_dict saved to {Vars.similarity_mapping_pickle_eval}")


    # evaluate surrogate model
    if get_bb_output:
        surrogate_dataset_eval = ImagenetSurrogate(data_dict, pred_dict)
    else:
        surrogate_dataset_eval = ImagenetSurrogateLabels(data_dict)

    surrogate_batchsize = eval_vars.eval_batchsize
    surrogate_loader_eval = torch.utils.data.DataLoader(surrogate_dataset_eval, batch_size=surrogate_batchsize, shuffle=True)

    pred_model.eval()

    act_acc = 0
    total = 0

    n = len(Vars.class_codes)
    acc_matrix = np.zeros((n,n))


    class_ids = {l: i for i, l in enumerate(Vars.class_codes)}

    for data in surrogate_loader_eval:
        inputs, _, _, og_paths = data
        print(f"{inputs.size()}")

        outputs = pred_model(inputs) # size = 400, thus sim_mapping is from n_clusters = 50
        # model from run_dir, sim_mapping from kmeans_concept_dict from Vars.kmeans_conceptpickle_full with Vars from run_dir

        for y, p in zip(outputs, og_paths):
            total += 1

            l = p.split('/')[0]
            if torch.argmax(y).item() == class_ids[l]:
                act_acc += 1
            
            acc_matrix[class_ids[l]][torch.argmax(y).item()] += 1

    print(f"{act_acc/total=}")
    print(f"{acc_matrix=}")

    return act_acc/total, acc_matrix