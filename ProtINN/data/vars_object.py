import os
from configparser import ConfigParser, ExtendedInterpolation
import json
import random
from datetime import datetime
import shutil
import torch

class VarsObject:
    def __init__(self, cf_file, runtype):
        # init config file
        config = ConfigParser(interpolation=ExtendedInterpolation(), allow_no_value=True)
        self.cf_file = cf_file
        config.read(cf_file)

        # dataset
        dataset = config['dataset']
        self.get_classes = dataset.get('get_classes')
        self.class_index_file = dataset.get('class_index_file')
        with open(self.class_index_file) as f:
            self.imagenet_classes = json.load(f)

        self.class_list_dir = dataset.get('class_list_dir')
        if self.get_classes == 'manual':
            self.class_codes = [i for i in dataset.get('manual_class_codes').split('\n') if i]
            self.class_list_file = dataset.get('class_list_save_file')
            with open(os.path.join(self.class_list_dir, self.class_list_file), 'w') as f:
                json.dump(self.class_codes, f, indent=2)
            self.class_list_conf = '-'+self.class_list_file.replace('.txt','')
        elif self.get_classes == 'random':
            n_random_classes = dataset.getint('n_random_classes')
            self.class_codes = selected_classes = random.sample(list(self.imagenet_classes.keys()), n_random_classes)
            self.class_list_file = dataset.get('class_list_save_file')
            with open(os.path.join(self.class_list_dir, self.class_list_file), 'w') as f:
                json.dump(self.class_codes, f, indent=2)
            self.class_list_conf = '-'+self.class_list_file.replace('.txt','')
        elif self.get_classes == 'from_file':
            self.class_list_file = dataset.get('class_list_load_file')
            with open(os.path.join(self.class_list_dir, self.class_list_file), 'r') as f:
                self.class_codes = json.load(f)
            self.class_list_conf = '-'+self.class_list_file.replace('.txt','')
        else:
            print(f"incorrect value for get_classes argument")
        print(self.class_codes)

        self.n_segs = [int(i) for i in dataset.get('n_segs').split('\n') if i]
        self.n_string = '-'.join(map(str, self.n_segs))
        self.bg_mode = dataset.get('bg_mode')
        self.data_folder_root = dataset.get('data_folder_root')

        # segmentation
        segmentation = config['segmentation']
        self.segment_images = segmentation.getboolean('segment_images')
        self.folder_to_segment = segmentation.get('folder_to_segment')

        # create folder for saving run details
        start_time = datetime.now()
        dt_str = '_'.join(str(start_time).split(' ')).split('.')[0]
        self.settings_dir = f"{dt_str}_{len(self.class_codes)}-classes_{self.bg_mode}_{runtype}"
        self.settings_path = os.path.join(self.class_list_dir, self.settings_dir)
        if not os.path.exists(self.settings_path):
            os.mkdir(self.settings_path)
        with open(os.path.join(self.settings_path, 'class_codes.json'), 'w') as f:
            json.dump(self.class_codes, f, indent=2)
        shutil.copy(cf_file, self.settings_path)

        # training
        training = config['training']
        # self.train_folder = training.get('train_folder_root')
        self.train_batchsize = training.getint('batchsize')
        # self.class_loss_scaling = training.getint('class_loss_scaling')
        # self.concept_loss_scaling = training.getint('concept_loss_scaling')
        # self.train_imgs_per_class = training.getint('max_train_imgs_per_class')
        self.surrogate_target = training.get('surrogate_target')
        self.weight_for_bb = training.getfloat('weight_for_bb')
        self.n_epochs = training.getint('n_epochs')
        self.max_epochs = training.getint('max_epochs')
        self.checkpoint_dir = training.get('checkpoint_savedir')

        # model
        self.model_path = config['model'].get('model_path')

        # cuda
        self.cuda_nr = config['cuda'].getint('cuda_nr')
        self.cuda_dev = torch.device(f"cuda:{self.cuda_nr}")

        # kmeans clustering
        kmeans = config['kmeans_clustering']
        self.min_concept_size = kmeans.getint('min_concept_size')
        self.max_concept_size = kmeans.getint('max_concept_size')
        if self.max_concept_size == 0:
            self.max_concept_size = None
        self.n_clusters_kmeans = kmeans.getint('n_clusters')
        self.save_concepts_kmeans = kmeans.getboolean('save_concepts')

        # evaluation
        evaluation = config['evaluation']
        # self.val_folder_root = evaluation.get('val_folder_root')
        # self.path_base = evaluation.get('path_base')
        self.eval_batchsize = evaluation.getint('eval_batchsize')
        # self.save_segs_eval = evaluation.getboolean('save_segs') # remove?

        # pickles and other filenames
        folders = config['folders']
        #self.old_pickle_date = folders.get('old_pickle_date') # added for more flexibility in old vs new pickle files (e.g. keep old activations, create new weights)
        #self.pickle_date = folders.get('pickle_date')
        self.pickle_folder = folders.get('pickle_folder')
        self.concept_folder = folders.get('concept_folder')

        today = ''.join(dt_str.split('-')).split('_')[0]
        self.acts_id = folders.get('activation_pickle_id')
        if self.acts_id is '':
            self.acts_id = today
        self.concept_id = folders.get('concept_pickle_id')
        if self.concept_id is '':
            self.concept_id = today
        self.sim_id = folders.get('similarity_pickle_id')
        if self.sim_id is '':
            self.sim_id = today
        self.ibinn_id = folders.get('bb_output_pickle_id')
        if self.ibinn_id is '':
            self.ibinn_id = today
        self.surrogate_id = folders.get('pred_model_pickle_id')
        if self.surrogate_id is '':
            self.surrogate_id = today

        # # TODO --> THESE 3 USED TO HAVE old_pickle_date
        # self.acts_pickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_activations_{self.old_pickle_date}.pickle.compressed"
        self.acts_dict_pickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_acts-dict_{self.acts_id}.pickle.compressed"
        self.acts_dict_pickle_eval = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_acts-dict-eval_{self.acts_id}.pickle.compressed"

        self.kmeans_centerpickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-concept-centers_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.concept_id}.pickle.compressed"
        # self.kmeans_conceptpickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-concept-dict_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.pickle_date}.pickle.compressed"
        self.kmeans_conceptpickle_full = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-full-concept-dict_{self.min_concept_size}-{self.n_clusters_kmeans}_{self.concept_id}.pickle.compressed"
        # self.kmeans_freqweightpickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-freq-weights_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.pickle_date}.pickle.compressed"
        self.acts_dict_pickle_small = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_acts-dict-small_{self.min_concept_size}-{self.n_clusters_kmeans}_{self.acts_id}_{self.concept_id}.pickle.compressed"
        # # added to test:
        # self.kmeans_percweightpickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-perc-weights_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.pickle_date}.pickle.compressed"
        # self.kmeans_lossweightpickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-loss-weights_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.pickle_date}.pickle.compressed"
        # self.kmeans_lossweightedpickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-loss-weighted-{self.concept_loss_scaling}sconc-{self.class_loss_scaling}sclass_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.pickle_date}.pickle.compressed"
        # self.kmeans_tfidfweightpickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-tfidf-weights_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.pickle_date}.pickle.compressed" #HAD -v3
        self.concepts_dir_kmeans = f"{self.concept_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-concepts_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.concept_id}"

        # self.linkagepickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_hac-dendrogram_{self.n_clusters_hac}-{self.linkage_type}-{self.distance}_{self.pickle_date}.pickle.compressed"
        # self.hac_conceptpickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_hac-concept-dict_{self.n_clusters_hac}-{self.linkage_type}-{self.distance}_{self.pickle_date}.pickle.compressed"
        # self.center_dict_pickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_hac-center-dict_{self.n_clusters_hac}-{self.linkage_type}-{self.distance}_{self.pickle_date}.pickle.compressed"
        # self.dendrogram_file = f"{self.plot_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_hac-dendrogram_{self.n_clusters_hac}-{self.linkage_type}-{self.distance}_{self.pickle_date}.JPEG"
        # self.hac_plotname = f"{self.plot_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_hac-2d-plot_{self.n_clusters_hac}-{self.linkage_type}-{self.distance}_{self.pickle_date}.JPEG"
        # self.concepts_dir_hac = f"{self.concept_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_hac-concepts_{self.n_clusters_hac}-{self.linkage_type}-{self.distance}_{self.pickle_date}"

        self.similarity_mapping_pickle = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-normalised-similarity-mapping_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.sim_id}.pickle.compressed" #HAD -v2
        self.similarity_mapping_pickle_eval = f"{self.pickle_folder}/{self.bg_mode}-segments_n-{self.n_string}_{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_kmeans-normalised-similarity-mapping-eval_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.sim_id}.pickle.compressed" #HAD -v2
        self.ibinn_trainoutput_pickle = f"{self.pickle_folder}/{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_ibinn-normalized-train-output_{self.ibinn_id}.pickle.compressed" #HAD -v2
        self.ibinn_evaloutput_pickle = f"{self.pickle_folder}/{len(self.class_codes)}-classes{self.class_list_conf}_beta-8_ibinn-normalized-eval-output_{self.ibinn_id}.pickle.compressed"

        self.target_type = self.surrogate_target
        if self.surrogate_target == 'weighted':
            self.target_type = f"weighted_{self.weight_for_bb}-{1-self.weight_for_bb}"
        self.epoch_str = "conv" if self.n_epochs==0 else f"{self.n_epochs}"
        self.checkpoint_name = f"ProtoCeptron-{self.bg_mode}-{len(self.class_codes)}classes-{self.target_type}-{self.epoch_str}-{self.surrogate_id}.pt"

        self.surrogate_checkpoint = f'{self.checkpoint_dir}/{self.checkpoint_name}'

        #self.eval_heatmap_dir = f"{self.bg_mode}_{self.n_string}_{len(self.class_codes)}_{self.class_list_conf}_beta8_{self.min_concept_size}-{self.max_concept_size}-{self.n_clusters_kmeans}_{self.target_type}_{self.n_epochs}_{self.surrogate_id}"


    def classification_objects(self, cl_file):
        cl_config = ConfigParser(interpolation=ExtendedInterpolation())
        self.cf_file = cl_file
        cl_config.read(cl_file)

        # classification
        images = cl_config['images']
        self.img_paths = [i for i in images.get('img_paths').split('\n') if i]

        return self.img_paths

    def evaluation_object(self, cl_file):
        eval_vars = Eval_vars(cl_file)

        return eval_vars


class Eval_vars:
    def __init__(self, cl_file):
        cl_config = ConfigParser(interpolation=ExtendedInterpolation())
        self.cf_file = cl_file
        cl_config.read(cl_file)

        # classification
        evaluation = cl_config['evaluation']
        self.eval_batchsize = evaluation.getint('eval_batchsize')
        self.segment_images = evaluation.getboolean('segment_images')
        self.folder_to_segment = evaluation.get('folder_to_segment')
        self.data_folder_root = evaluation.get('data_folder_root')
