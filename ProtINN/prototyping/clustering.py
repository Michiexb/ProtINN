import sklearn.cluster as skcluster
import numpy as np
import bz2
import pickle
import sys
import os
import shutil


sys.setrecursionlimit(100000)


def split_list(ls, n):
    if n > 1:
        s = int(len(ls) / n) + (len(ls) % n > 0)
        sub_ls = []
        print(f"split list pre append")
        for i in range(n):
            sub_ls.append(ls[i*s:(i+1)*s])
        print(f"split list post append")
    else:
        sub_ls = [ls]
    return sub_ls



def cluster(acts, n_clusters):
    """Runs unsupervised clustering algorithm on activations with k-means clustering method.

    Args:
        acts: The activation vectors of the datapoints/images.
        n_clusters: The amount of clusters to be created.

    Returns:
        label: The cluster assignment label of each data point.
        cost: The clustering cost of each data point.
        centers: The cluster centers."""

    km = skcluster.MiniBatchKMeans(n_clusters, random_state=0, batch_size=384)

    sub_acts = split_list(acts, 10)#int(len(acts)/(vars.n_clusters_kmeans+5)))
    for sub in sub_acts:
        km = km.partial_fit(sub)
    centers = km.cluster_centers_
    print(f"cluster fitting done")

    # SPLIT TO AVOID MEMORY FULL
    sub_acts2 = split_list(acts, int(len(acts)/10000)) #10000 is #images/act_groups per sub list
    asg, cost = [], []
    for a in sub_acts2:
        center_np = np.expand_dims(centers,0)
        subacts_np = np.expand_dims(a,1)
        d = np.linalg.norm(subacts_np - center_np, ord=1, axis=-1)
        asg.extend(np.argmin(d, -1))
        cost.extend(np.min(d, -1))
    print(f"cost calculation done")

    asg = np.array(asg)
    cost = np.array(cost)

    return asg, cost, centers


def create_concepts(acts_dict, min_concept_size, max_concept_size, n_clusters, centerpickle):
    # img_paths, activations
    img_paths = list(acts_dict.keys())
    activations = [acts_dict[p] for p in img_paths]

    """creates concepts from the image segments"""
    image_numbers = []
    for p in img_paths:
        og_file = '_'.join(p.split('_')[:-1]) + '.JPEG'
        image_numbers.append(og_file)

    concept_dict = {}

    if os.path.exists(centerpickle):
        with bz2.BZ2File(centerpickle, 'r') as f:
            concept_dict['label'], concept_dict['cost'], centers = pickle.load(f)
        print(f"concept_dict loaded from {centerpickle}")

    else:
        concept_dict['label'], concept_dict['cost'], centers = cluster(activations, n_clusters)
        
        with bz2.BZ2File(centerpickle, 'w') as f:
            pickle.dump([concept_dict['label'], concept_dict['cost'], centers], f)
        print(f"concept_dict saved to {centerpickle}")

    path_array = np.array(img_paths)
    image_numbers = np.array(image_numbers)

    concept_number, concept_dict['concepts'] = 0, []
    for i in range(concept_dict['label'].max() + 1):
        label_idxs = np.where(concept_dict['label'] == i)[0]

        if len(label_idxs) > min_concept_size:
            concept_costs = concept_dict['cost'][label_idxs]
            concept_idxs = label_idxs[np.argsort(concept_costs)[:max_concept_size]]

            concept_number += 1
            concept = 'concept{}'.format(concept_number)
            concept_dict['concepts'].append(concept)
        
            costs = [concept_dict['cost'][i] for i in concept_idxs]

            concept_dict[concept] = {'image_paths': path_array[concept_idxs], 'concept_costs': costs}

            concept_dict[concept + '_center'] = centers[i]

    # concept_dict['image_classes'] = class_dict
    concept_dict.pop('label', None)
    concept_dict.pop('cost', None)
    return concept_dict



def save_concepts(concept_dict, output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for c in concept_dict['concepts']:
        concept_folder = os.path.join(output_folder, c)
        if not os.path.isdir(concept_folder):
            os.mkdir(concept_folder)
        c_dict = concept_dict[c]
        for file in c_dict['image_paths']:
            split_path = file.split('/')
            # print(split_path)
            cls = split_path[3]
            img_name = split_path[4]
            new_name = f"{cls}_{img_name}"
            shutil.copyfile(file, os.path.join(concept_folder, new_name))