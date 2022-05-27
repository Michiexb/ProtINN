import os
import torch

def matrix_sim_mapping(acts_dict, kmeans_concept_dict, acts_dict_train, Vars):
    # create data_dict to return
    data_dict = {'image_paths': [], 'images': [], 'similarity': [], 'segment_paths': []}

    img_paths = list(acts_dict.keys())
    img_dict = {path: f"{'_'.join(path.split('_')[:-1])}.JPEG" for path in img_paths}
    # img_dict: segname: last part of filename

    og_img_dict = {}
    for k, v in img_dict.items():
        if v not in og_img_dict.keys():
            og_img_dict[v] = [k]
        else:
            og_img_dict[v].append(k)
    data_dict['image_paths'] = list(og_img_dict.keys())

    # because of different numbers of segments per image:
    n_segs_p_img = [len(og_img_dict[key]) for key in data_dict['image_paths']]
    imgs_per_len = {l: [] for l in set(n_segs_p_img)}
    for l, img in zip(n_segs_p_img, data_dict['image_paths']):
        imgs_per_len[l].append(img)
    img_order = sum(imgs_per_len.values(), [])

    all_segs = [torch.Tensor([[acts_dict[seg] for seg in og_img_dict[og]] for og in imgs_per_len[lenn]]) for lenn in imgs_per_len.keys()]
    
    # GRABBING CLUSTER MEDOIDS:
    concepts = torch.Tensor([acts_dict_train[kmeans_concept_dict[conc]['image_paths'][0]] for conc in kmeans_concept_dict['concepts']])
    # WHEN USING CLUSTER CENTROID, REPLACE WITH:
    # concepts = torch.Tensor([kmeans_concept_dict[c+'_center'] for c in kmeans_concept_dict['concepts']])
    

    # furthest segments as Tensor with shape [[activations of furthest segment in concept 0],...,[...in concept n]]
    furthest_segs = torch.Tensor([acts_dict_train[kmeans_concept_dict[conc]['image_paths'][-1]] for conc in kmeans_concept_dict['concepts']])

    # calculating the euclidean (p=2) distance between concepts and segments
    dists_list = [torch.cdist(concepts, segments, p=2) for segments in all_segs]
    closest_list = [torch.min(dists, dim=2) for dists in dists_list]
    # for each image, find closest segments per concept (returns values and indices)
    mindists = torch.stack([cv for clos in closest_list for cv in clos.values])
    closest_segs = torch.stack([ci for clos in closest_list for ci in clos.indices])
    # distance between furthest segs and concept_center (now matrix calculation, but needed: identity only)
    furthest_dist_matrix = torch.cdist(concepts, furthest_segs, p=2)
    # taking only identity values of furthest_dist_matrix
    furthest_dist_list = [furthest_dist_matrix[i][i] for i in range(len(furthest_segs))]
    furthest_dist = torch.Tensor(furthest_dist_list)
    # distance to similarity
    sims = -abs(mindists)/((furthest_dist/10)+furthest_dist)
    # similarity normalization (inside boundary --> 1, outside boundary --> 0)
    norms = torch.tanh(sims)+1
    # add to data_dict
    data_dict['similarity'] = norms

    # for each image, a list with for each concept the filepath of the closest segment of given image
    segpaths = [[og_img_dict[og_img_id][i] for i in og_seg_ids] for og_img_id, og_seg_ids in zip(img_order, closest_segs)]
    # add to data_dict
    data_dict['image_paths'] = [os.path.join(Vars.data_folder_root,'/'.join(im.split('/')[-3:])) for im in img_order] # val or train / n-class missing
    data_dict['segment_paths'] = segpaths
    return data_dict
