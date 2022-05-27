import matplotlib.pyplot as plt
import numpy as np
import json
import os


def make_heatmap(matrix, matrix_name, Vars):
    n = len(Vars.class_codes)

    # MAKE HEATMAP
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # Setting the labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    # labeling respective list entries
    with open('ProtINN/data/imagenet_classcode_to_index.json') as f:
        class_dict = json.load(f)
    class_names = [class_dict[key][1] for key in Vars.class_codes]
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Creating text annotations by using for loop
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, int(matrix[i, j]),
                        ha="center", va="center", color="w", fontsize="xx-small")

    ax.set_title(matrix_name)
    fig.tight_layout()

    #n_epochs = vars.n_epochs
    #target_type = vars.surrogate_target
    #if vars.surrogate_target == 'weighted':
    #    target_type = f"weighted_{int(vars.weight_for_bb*100)}{int((1-vars.weight_for_bb)*100)}"
    # visdir = f"{vars.bg_mode}_wrt_{target_type}_{n_epochs}"
    #visdir = f"{vars.bg_mode}_{vars.n_string}_{len(vars.class_codes)}_{vars.class_list_conf}_beta8_{vars.min_concept_size}-{vars.max_concept_size}-{vars.n_clusters_kmeans}_{vars.pickle_date}_{target_type}_{n_epochs}"
    
    savedir = Vars.settings_path #f"Data/visualisations/heatmaps/{visdir}"
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    fig.savefig(f"{savedir}/{matrix_name}_{len(Vars.class_codes)}-classes-{Vars.bg_mode}-{Vars.surrogate_id}.PNG")