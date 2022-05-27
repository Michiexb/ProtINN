# IMPORT MODULES
# base modules needed
import os
import bz2
import pickle
import torch
from datetime import datetime
import argparse
import logging

# trained model
from ibinn_imagenet.model.classifiers.invertible_imagenet_classifier import trustworthy_gc_beta_8

# needed functions
from ProtINN.segmentation.segment_test_img import segment_folder_imgs
from ProtINN.model.activations import get_activations
from ProtINN.prototyping.clustering import create_concepts, save_concepts
from ProtINN.data.surrogate_target import create_targets
from ProtINN.classification.similarity_mapping import matrix_sim_mapping
from ProtINN.model.surrogate import training_step

# needed classes
from ProtINN.data.dataset import ImagenetSegments, ImagenetTarget, ImagenetSurrogate, ImagenetSurrogateLabels
from ProtINN.model.surrogate import PredictionLayer
from ProtINN.data.vars_object import VarsObject


'''How to run:
python training.py config_file_path
'''

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
    Vars = VarsObject(args.config_file, 'train')

    logging.basicConfig(filename=f"{Vars.settings_path}/output.txt", level=logging.DEBUG, format='')
    logging.info(f"{args.config_file=}")
    print(f"time for setup: {datetime.now()-current_time}")
    logging.info(f"time for setup: {datetime.now()-current_time}")
    current_time = datetime.now()

    # LOAD DATA
    # segment data if desired
    # segment images in Vars.folder_to_segment and save segments to Vars.data_folder_root/train
    # segment images into Vars.n_segs number of segments with Vars.bg_mode as background mode (grey, blurred or patch)
    if Vars.segment_images:
        segment_folder_imgs(Vars.class_codes, Vars.folder_to_segment, Vars.data_folder_root, 'train', Vars.n_segs, Vars.bg_mode)
        print(f"time for segmenting images: {datetime.now()-current_time}")
        logging.info(f"time for segmenting images: {datetime.now()-current_time}")
        current_time = datetime.now()

    # load segments in Vars.data_folder_root/train directory to dataset and dataloader
    train_folder = os.path.join(Vars.data_folder_root, 'train')
    data = ImagenetSegments(train_folder, Vars, Vars.train_batchsize)
    data_loader = data.data_loader
    print(f"time for loading segment dataset: {datetime.now()-current_time}")
    logging.info(f"time for loading segment dataset: {datetime.now()-current_time}")
    current_time = datetime.now()


    get_bb_output = Vars.surrogate_target in ['weighted', 'multiplied', 'blackbox']
    if get_bb_output:
        # GET FULL IMAGE BLACKBOX PREDICTIONS FOR MODEL TARGET
        pred_pickle = Vars.ibinn_trainoutput_pickle
        if os.path.exists(pred_pickle):
            with bz2.BZ2File(pred_pickle, 'r') as f:
                pred_dict = pickle.load(f)
            print(f"{pred_pickle} loaded")
            logging.info(f"{pred_pickle} loaded")
        else:
            beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
            beta_8_model = beta_8_model.to(Vars.cuda_dev)
            beta_8_model.eval()

            target_dataset = ImagenetTarget(os.path.join(Vars.folder_to_segment,'train'), Vars, 50)
            target_train_loader = torch.utils.data.DataLoader(target_dataset, batch_size=target_dataset.batch_size, shuffle=True, num_workers=12, pin_memory=False, sampler=None)

            class_ids = [int(Vars.imagenet_classes[c][0]) for c in Vars.class_codes]
            pred_dict = create_targets(target_train_loader, beta_8_model, class_ids)

            with bz2.BZ2File(pred_pickle, 'w') as f:
                pickle.dump(pred_dict, f)
            print(f"pred_dict saved to {pred_pickle}")
            logging.info(f"pred_dict saved to {pred_pickle}")

        print(f"time for creating/loading target dataset: {datetime.now()-current_time}")
        logging.info(f"time for creating/loading target dataset: {datetime.now()-current_time}")
        current_time = datetime.now()

    # GET SIMILARITY MAPPING
    if not os.path.exists(Vars.similarity_mapping_pickle):
        # GET ACTIVATIONS OF SEGMENTS
        # if done before, load:
        if os.path.exists(Vars.acts_dict_pickle):
            with bz2.BZ2File(Vars.acts_dict_pickle, 'r') as f:
                acts_dict = pickle.load(f)
            print(f"{Vars.acts_dict_pickle} loaded")
            logging.info(f"{Vars.acts_dict_pickle} loaded")

        # otherwise, get actications and save to dictionary:
        else:
            # load model
            beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
            model = beta_8_model.model

            acts_fc, _, img_paths = get_activations(data_loader, model, Vars.cuda_dev)
            acts_dict = {}
            for img, a_fc in zip(img_paths, acts_fc):
                acts_dict[img] = a_fc

            with bz2.BZ2File(Vars.acts_dict_pickle, 'w') as f:
                pickle.dump(acts_dict, f)
            print(f"acts_dict saved to {Vars.acts_dict_pickle}")
            logging.info(f"acts_dict saved to {Vars.acts_dict_pickle}")

            # TODO: release acts_fc memory ?
            del(acts_fc)

        print(f"time for creating/loading segment activations: {datetime.now()-current_time}")
        logging.info(f"time for creating/loading segment activations: {datetime.now()-current_time}")
        current_time = datetime.now()



    # CREATE / LOAD PROTOTYPES
    if os.path.exists(Vars.kmeans_conceptpickle_full):
        with bz2.BZ2File(Vars.kmeans_conceptpickle_full, 'r') as f:
            kmeans_concept_dict = pickle.load(f)
        print(f"{Vars.kmeans_conceptpickle_full} loaded")
        logging.info(f"{Vars.kmeans_conceptpickle_full} loaded")
    else:
        kmeans_concept_dict = create_concepts(acts_dict, Vars.min_concept_size, Vars.max_concept_size, Vars.n_clusters_kmeans, Vars.kmeans_centerpickle)

        with bz2.BZ2File(Vars.kmeans_conceptpickle_full, 'w') as f:
            pickle.dump(kmeans_concept_dict, f)
        print(f"kmeans_concept_dict saved to {Vars.kmeans_conceptpickle_full}")
        logging.info(f"kmeans_concept_dict saved to {Vars.kmeans_conceptpickle_full}")

    print(f"time for creating/loading concepts: {datetime.now()-current_time}")
    logging.info(f"time for creating/loading concepts: {datetime.now()-current_time}")
    current_time = datetime.now()


    if Vars.save_concepts_kmeans:
        save_concepts(kmeans_concept_dict, Vars.concepts_dir_kmeans)
        
        print(f"time for saving concept center to files: {datetime.now()-current_time}")
        logging.info(f"time for saving concept center to files: {datetime.now()-current_time}")
        current_time = datetime.now()



    if not os.path.exists(Vars.acts_dict_pickle_small):
        pths = [kmeans_concept_dict[c]['image_paths'][i] for c in kmeans_concept_dict['concepts'] for i in [0, -1]]
        acts_dict_small = {key: acts_dict[key] for key in pths}
        
        with bz2.BZ2File(Vars.acts_dict_pickle_small, 'w') as f:
            pickle.dump(acts_dict_small, f)
        print(f"acts_dict_small saved to {Vars.acts_dict_pickle_small}")
        logging.info(f"acts_dict_small saved to {Vars.acts_dict_pickle_small}")

        print(f"time for creating smaller acts_dict: {datetime.now()-current_time}")
        logging.info(f"time for creating smaller acts_dict: {datetime.now()-current_time}")
        current_time = datetime.now()


    # get normalized similarity scores
    # if sim-scores not calculated -> calculate -> save
    # else -> load sim-scores
    if os.path.exists(Vars.similarity_mapping_pickle):
        with bz2.BZ2File(Vars.similarity_mapping_pickle, 'r') as f:
            data_dict = pickle.load(f)
        print(f"{Vars.similarity_mapping_pickle} loaded")
        logging.info(f"{Vars.similarity_mapping_pickle} loaded")
    else:
        data_dict = matrix_sim_mapping(acts_dict, kmeans_concept_dict, acts_dict, Vars)
        
        with bz2.BZ2File(Vars.similarity_mapping_pickle, 'w') as f:
            pickle.dump(data_dict, f)
        print(f"data_dict saved to {Vars.similarity_mapping_pickle}")
        logging.info(f"data_dict saved to {Vars.similarity_mapping_pickle}")

    print(f"time for creating/loading similarity mapping: {datetime.now()-current_time}")
    logging.info(f"time for creating/loading similarity mapping: {datetime.now()-current_time}")
    current_time = datetime.now()


    # TRAIN PREDICTION MODEL
    # load model shape and data
    if get_bb_output:
        dataset_train = ImagenetSurrogate(data_dict, pred_dict)
    else:
        dataset_train = ImagenetSurrogateLabels(data_dict)
    pred_layer = PredictionLayer(len(kmeans_concept_dict['concepts']), len(Vars.class_codes))

    if Vars.n_epochs==0:
        if os.path.exists(Vars.similarity_mapping_pickle_eval):
            # load data_dict_eval
            with bz2.BZ2File(Vars.similarity_mapping_pickle_eval, 'r') as f:
                data_dict_eval = pickle.load(f)
            print(f"{Vars.similarity_mapping_pickle_eval} loaded")
            logging.info(f"{Vars.similarity_mapping_pickle_eval} loaded")
        else:
            # create data_dict_eval
            val_folder = os.path.join(Vars.data_folder_root, 'val')
            data_eval = ImagenetSegments(val_folder, Vars, Vars.eval_batchsize)
            data_loader_eval = data_eval.data_loader
            
            if not "model" in globals() or not "model" in locals():
                # load model
                beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
                model = beta_8_model.model
            acts_fc, _, img_paths = get_activations(data_loader_eval, model, Vars.cuda_dev)
            acts_dict_eval = {}
            for img, a_fc in zip(img_paths, acts_fc):
                acts_dict_eval[img] = a_fc
            with bz2.BZ2File(Vars.acts_dict_pickle_eval, 'w') as f:
                pickle.dump(acts_dict_eval, f)
            print(f"acts_dict_eval saved to {Vars.acts_dict_pickle_eval}")

            # TODO: release acts_fc memory
            del(acts_fc)

            if not "acts_dict_small" in globals() or not "acts_dict_small" in locals():
                with bz2.BZ2File(Vars.acts_dict_pickle_small, 'r') as f:
                    acts_dict_small = pickle.load(f)
                print(f"{Vars.acts_dict_pickle_small} loaded")
                logging.info(f"{Vars.acts_dict_pickle_small} loaded")
            data_dict_eval = matrix_sim_mapping(acts_dict_eval, kmeans_concept_dict, acts_dict_small, Vars)

        if get_bb_output:
            if os.path.exists(Vars.ibinn_evaloutput_pickle):
                # load pred_dict_eval
                with bz2.BZ2File(Vars.ibinn_evaloutput_pickle, 'r') as f:
                    pred_dict_eval = pickle.load(f)
                print(f"{Vars.ibinn_evaloutput_pickle} loaded")
                logging.info(f"{Vars.ibinn_evaloutput_pickle} loaded")
            else:
                # create pred_dict_eval
                beta_8_model = trustworthy_gc_beta_8(pretrained=True, pretrained_model_path = Vars.model_path)
                beta_8_model = beta_8_model.to('cuda:0')
                beta_8_model.eval()

                target_dataset_eval = ImagenetTarget(os.path.join(Vars.folder_to_segment,'val'), Vars, 50)
                target_loader_eval = torch.utils.data.DataLoader(target_dataset_eval, batch_size=target_dataset.batch_size, shuffle=True, num_workers=12, pin_memory=False, sampler=None)

                class_ids = [int(Vars.imagenet_classes[c][0]) for c in Vars.class_codes]
                pred_dict_eval = create_targets(target_loader_eval, beta_8_model, class_ids)

            # train prediction model until convergence
            dataset_eval = ImagenetSurrogate(data_dict_eval, pred_dict_eval)
        else:
            dataset_eval = ImagenetSurrogateLabels(data_dict_eval)

        pred_layer = training_step(pred_layer, dataset_train, Vars, dataset_eval)
    else:
        # train prediction model until Vars.epochs
        pred_layer = training_step(pred_layer, dataset_train, Vars)



    torch.save(pred_layer.state_dict(), Vars.surrogate_checkpoint)
    torch.save(pred_layer.state_dict(), os.path.join(Vars.settings_path, Vars.checkpoint_name))

    print(f"time for entire run: {datetime.now()-start_time}")
    logging.info(f"time for entire run: {datetime.now()-start_time}")


if __name__ == '__main__':
    func()