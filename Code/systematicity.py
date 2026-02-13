import pandas as pd
import itertools
from tqdm import tqdm
from information_theory import mi
from partition_manipulation import canonical_label
from deixis import enumerate_possible_lexicons


def generate_all_possible_partitions(features):
    num_features = len(features)
    possible_partitions = []
    for i in range(1, num_features + 1):
        partitions_and_encoders = list(enumerate_possible_lexicons(num_features, i))
        partitions = [val[0] for val in partitions_and_encoders]
        possible_partitions.extend(partitions)
    return possible_partitions


def generate_convex_partitions(features):

    num_features = len(features)
    num_list = [i for i in range(num_features)]

    def recursive_partitioning(curr_list, curr_depth):

        all_partitions = []
        base_partition = (curr_depth, ) * len(curr_list)
        all_partitions.append(base_partition)

        for i in range(1, len(curr_list)):
            remaining = curr_list[i:]
            sub_partitions = recursive_partitioning(remaining, curr_depth+1)
            for partition in sub_partitions:
                new_base_partition = (curr_depth, ) * i
                all_partitions.append(new_base_partition + partition)
        return all_partitions
    
    return recursive_partitioning(num_list, 0)


def empty_cluster_mapping(features):

    unique_clusters = list(range(len(features)))

    dimension_to_cluster = {}
    for feature in features:
        dimension_to_cluster[feature] = {cluster: 0 for cluster in unique_clusters}
    return dimension_to_cluster

 

def generate_all_clusterings(relations, only_convex=False):

    clusterings = {}

    for relation_name in relations:
        relation = relations[relation_name]
        features = relation.columns

        if only_convex:
            possible_partitions = generate_convex_partitions(features)
        else:
            possible_partitions = generate_all_possible_partitions(features)
            
            
        # Convert partitions (0, 0, 0) into conditional distributions p(z|d)
        mappings = {}

        for partition in possible_partitions:
            feature_to_cluster = empty_cluster_mapping(features)

            # Assign each feature value to a particular cluster
            for feature, cluster in zip(features, partition):
                feature_to_cluster[feature][cluster] = 1 

            mapping_df = pd.DataFrame(feature_to_cluster).T
            cluster_id = "".join(map(str, partition)) # converts (0, 0, 0) to string "000"
            mappings[cluster_id] = mapping_df

        clusterings[relation_name] = mappings

    return clusterings



def clustering_complexity(needs, relation, clustering):

    pm = needs[:, None, None]
    pdm = relation[:, :, None]
    pdcd = clustering[None, :, :]

    p_mddc = pm * pdm * pdcd
    p_mdc = p_mddc.sum(axis=0)
    return mi(p_mdc)



def clustering_informativity(needs, relation, clustering, encoder):

    pm = needs[:, None, None, None]
    pwm = encoder[:, None, :, None]
    pdm = relation[:, :, None, None]
    pdcd = clustering[None, :, None, :]

    p_mdwdc = pm * pwm * pdm * pdcd
    p_wdc = p_mdwdc.sum(axis=(0, 1))
    return mi(p_wdc)



def mini_systematicity_scoring(data_loader, language, prior, relations, clusterings, combo_of_interest):
    encoder = data_loader.language_to_encoder[language]
    relation_names = list(relations.keys())

    complexities_list = []
    informativities_list = []

    for j in range(len(combo_of_interest)):

        dim = relation_names[j]
        relation = relations[dim]
        clustering = clusterings[dim][combo_of_interest[j]]

        complexity = clustering_complexity(prior, relation, clustering)
        informativity = clustering_informativity(prior, relation, clustering, encoder)
        complexities_list.append(complexity)
        informativities_list.append(informativity)

    row = [language] + list(combo_of_interest) + complexities_list + informativities_list
    all_scores = [row]
    
    cluster_cols = [f'{rel}_cluster' for rel in relation_names]
    complexity_cols = [f'i_{rel}' for rel in relation_names] 
    informativity_cols = [f'i_w_{rel}' for rel in relation_names]
    
    columns = ['language'] + cluster_cols + complexity_cols + informativity_cols
    all_scores = pd.DataFrame(all_scores, columns=columns)
    all_scores = all_scores.set_index("language")

    all_scores['sys_complexity'] = all_scores[[f'i_{rel}' for rel in relation_names]].sum(axis=1)
    all_scores['sys_informativity'] = all_scores[[f'i_w_{rel}' for rel in relation_names]].sum(axis=1)
    all_scores = all_scores.round(6)

    return all_scores
    

def systematicity_scoring(data_loader, langs_of_interest, prior, relations, clusterings, combos_of_interest=None):
    """
    Note that the prior must be a 1-dimensional vector with shape (n, ), 
    where n is the number of meanings.
    """

    encoders = data_loader.language_to_encoder

    complexities = {dimension: {} for dimension in relations}
    informativities = {dimension: {cluster_id: {} for cluster_id in clusterings[dimension]} for dimension in relations}

    for dimension in relations:
        relation = relations[dimension]

        for cluster_id in clusterings[dimension]:
            clustering = clusterings[dimension][cluster_id]

            # Compute complexity
            complexity = clustering_complexity(prior, relation, clustering)
            complexities[dimension][cluster_id] = complexity

            # Compute informativity for all languages
            for lang in langs_of_interest:
                encoder = encoders[lang]
                informativity = clustering_informativity(prior, relation, clustering, encoder)
                informativities[dimension][cluster_id][lang] = informativity
        
    all_scores = []
    relation_names = list(relations.keys())

    if combos_of_interest is not None:
        cluster_combinations = combos_of_interest
    else:
        cluster_id_lists = [list(complexities[dim].keys()) for dim in relation_names]
        cluster_combinations =  itertools.product(*cluster_id_lists)
    
    for cluster_combination in cluster_combinations:
        complexities_list = []
        informativities_list = []

        for i in range(len(relation_names)):
            complexities_list.append(complexities[relation_names[i]][cluster_combination[i]])
            informativities_list.append(informativities[relation_names[i]][cluster_combination[i]])
        
        for language in langs_of_interest:
            encoder_mi = [info_dict[language] for info_dict in informativities_list]
            row = [language] + list(cluster_combination) + complexities_list + encoder_mi
            all_scores.append(row)

    cluster_cols = [f'{rel}_cluster' for rel in relation_names]
    complexity_cols = [f'i_{rel}' for rel in relation_names] 
    informativity_cols = [f'i_w_{rel}' for rel in relation_names]
    
    columns = ['language'] + cluster_cols + complexity_cols + informativity_cols
    all_scores = pd.DataFrame(all_scores, columns=columns)
    all_scores = all_scores.set_index("language")

    all_scores['sys_complexity'] = all_scores[[f'i_{rel}' for rel in relation_names]].sum(axis=1)
    all_scores['sys_informativity'] = all_scores[[f'i_w_{rel}' for rel in relation_names]].sum(axis=1)
    all_scores = all_scores.round(6)

    return all_scores


def create_all_clustering_combinations(relations, clusterings, p_m, include_complexity=False):

    relation_names = list(relations.keys())
    cluster_id_lists = [list(clusterings[dim].keys()) for dim in clusterings]
    clustering_combinations = list(itertools.product(*cluster_id_lists))
    combo_complexities = []
    for i in tqdm(range(len(clustering_combinations))):
        combo = clustering_combinations[i]
        if include_complexity:
            combo_complexity = 0
            for j in range(len(combo)):
                dim = relation_names[j]
                combo_complexity += clustering_complexity(p_m, relations[dim], clusterings[dim][combo[j]])
        else:
            combo_complexity = 1
        combo_complexities.append((combo, combo_complexity))
    combo_queue = sorted(combo_complexities, key=lambda x: x[1], reverse=True)
    return combo_queue    

def identify_best_clustering_per_language(scores_df, gamma):

    diffs = scores_df['sys_complexity'] - gamma*scores_df['sys_informativity']
    diffs = diffs.reset_index()
    diffs.columns = ['language', 'diff']

    # For each language, only keep the row (i.e. clustering) where the language achieves its minimum score
    per_lang_diff = diffs.groupby(by='language')['diff'].idxmin()
    minimum_scores = scores_df.iloc[per_lang_diff]
    return minimum_scores
    

def optimal_partition_for_clusterings(cluster_combo):
    d1 = list(cluster_combo[1])
    d2 = list(cluster_combo[0])
    combos = itertools.product(d1, d2)
    words = list(map(lambda x: "".join(x), combos))
    partition = canonical_label(words)
    return partition
