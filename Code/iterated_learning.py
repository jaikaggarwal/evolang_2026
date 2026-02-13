import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os

from constants import CONFIG_DIR,  DATA_DIR, IMAGES_DIR, ALPHABET
from utils import serialize_object, load_serialized_object, make_directory
from config_template import load_yaml, ILConfig

from data_loader import load_data_loader
from systematicity import *
from vanilla_ib import compute_vanilla_ib_scores, make_curve
from objective_functions import compute_optimality
from visualization import visualize_tradeoff




class OptimalityParams:

    def __init__(self, tradeoff_signs, gammas, optimality_dir, images_dir):

        self.tradeoff_signs = tradeoff_signs
        self.gammas = gammas
        self.optimality_dir = optimality_dir
        self.images_dir = images_dir


def load_experiment_config(config_file_path: str):

    config_data = load_yaml(config_file_path)
    experiment_config = ILConfig(**config_data)

    # Add experiment name from file path
    l_index = config_file_path.rindex("/")
    r_index = config_file_path.rindex(".")
    experiment_name = config_file_path[l_index+1:r_index]
    experiment_config = experiment_config.model_copy(update={'experiment_name': experiment_name})
    return experiment_config


def set_up_directories(config):

    root_dir = f"{DATA_DIR}/{config.domain}/{config.experiment_name}/"
    systematicity_dir = f"{root_dir}/systematicity/"
    optimality_dir = f"{root_dir}/optimality/"
    images_dir = f"{IMAGES_DIR}/{config.domain}/{config.experiment_name}/"
    
    make_directory(root_dir)
    make_directory(systematicity_dir)
    make_directory(optimality_dir)
    make_directory(images_dir)
    make_directory(f"{images_dir}/vanilla_ib/")
    make_directory(f"{images_dir}/systematic_ib/")


    return systematicity_dir, optimality_dir, images_dir


def efficiency_by_chain(scores, base_scores):

    reformatted_scores = {}

    for chain_id in base_scores['chain'].unique():
        curr_chain = base_scores[base_scores['chain'] == chain_id]
        curr_optimality = scores.loc[curr_chain.index]
        x_axis = np.arange(curr_chain['generation'].max() + 1)
        y_axis = curr_optimality['normalized_diffs']
        reformatted_scores[chain_id] = dict(zip(x_axis, y_axis))
        
    reformatted_scores = pd.DataFrame(reformatted_scores).T
    reformatted_scores = reformatted_scores.ffill(axis=1)
    return reformatted_scores



def plot_trajectory(system, optimal_curve, fields, axis_labels, max_x=3, max_y=1.5, save_name=None):

    x_field = fields[0]
    y_field = fields[1]

    plt.plot(optimal_curve[x_field], optimal_curve[y_field], color="black")
    plt.plot(system[x_field], system[y_field])
    plt.plot(system[x_field].iloc[0], system[y_field].iloc[0], marker='o')
    plt.plot(system[x_field].iloc[-1], system[y_field].iloc[-1], marker='X', markersize=12)
    plt.fill_between(optimal_curve[x_field], optimal_curve[y_field]+0.01, max_y, color='grey', hatch="/", edgecolor="black")

    x_axis_label = axis_labels[0]
    y_axis_label = axis_labels[1]

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)

    if save_name is not None:
        plt.tight_layout()
        plt.savefig(save_name)
    else:
        plt.show()
    
    plt.clf()



def plot_deviation_from_optimality(deviation_df, save_name=None):

    x_axis = deviation_df.columns
    mean_line = deviation_df.mean()

    for chain_id in deviation_df.index:
        plt.plot(x_axis, deviation_df.loc[chain_id], c='grey', alpha=0.5)
    plt.plot(x_axis, mean_line)
    plt.xlabel("Generation", fontdict={"size": 15})
    plt.ylabel("Deviation from Optimality", fontdict={"size": 15})
    plt.tick_params(axis='both', which='major', labelsize=15)

    if save_name is not None:
        plt.tight_layout()
        plt.savefig(save_name)
    else:
        plt.show()
    
    plt.clf()



def visualize_facet_plot(vanilla_ib_deviation, systematic_ib_deviation, save_name=None):
    
    x_axis = vanilla_ib_deviation.columns

    plt.rcParams.update({'font.size': 30})
    fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(20, 14))
    fig.supxlabel("Generation", y=0.06)
    fig.supylabel("Deviation from Optimality")

    for chain_id in vanilla_ib_deviation.index:
        plt.subplot(4, 3, chain_id+1)
        plt.plot(x_axis, vanilla_ib_deviation.loc[chain_id], c='grey', alpha=0.5, label=r'$J_{IB}$')
        plt.plot(x_axis, systematic_ib_deviation.loc[chain_id], c='red', alpha=0.5, label=r'$J_{GIB}$')
        plt.title(f"Chain {ALPHABET[chain_id]}", fontdict={"size": 25})

    fig.legend([r'$J_{IB}$', r'$J_{GIB}$'], loc='upper right', bbox_to_anchor=(0.98, 0.95), ncol=1)

    if save_name is not None:
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()

    plt.rcdefaults()



def load_systematicity_information(data_loader, data_dir, p_m, relation_params, include_only_convex):
     
    print("Loading relations")
    relations = data_loader.load_relations(relation_params)

    print("Generating Clusterings")
    clusterings = generate_all_clusterings(relations, include_only_convex)

    relations = {key: relations[key].to_numpy() for key in relations}
    clusterings = {key: {cluster_id: clusterings[key][cluster_id].to_numpy() for cluster_id in clusterings[key]} for key in clusterings}

    print("Filtering 4-word Clustering Combinations")
    combo_queue_file = data_dir + f"4_word_combos_convex_{include_only_convex}.pkl"

    if os.path.exists(combo_queue_file):
        new_queue = load_serialized_object(combo_queue_file)
    else:
        combo_queue = create_all_clustering_combinations(relations, clusterings, p_m)
        new_queue = []
        for combo_pair in tqdm(combo_queue):
            combo = combo_pair[0]
            combo_partition = optimal_partition_for_clusterings((combo[1], combo[0]))
            if max(combo_partition) < 4:
                new_queue.append(combo)
        serialize_object(new_queue, combo_queue_file)

    print(f"Num clusterings: {len(new_queue)}")
    return relations, clusterings, new_queue



def score_systems(data_loader, data_dir, p_m, pu_m, relations, clusterings, combo_queue, inf_weight, beta):

    def extract_metadata(x, field):
        idx = 1 if field == 'chain' else 3
        split_name = x.name.split("_")
        return int(split_name[idx]) if x['ground_truth'] else -100
    
    vanilla_scores_file = data_dir + "vanilla_ib_scores.pkl"

    if os.path.exists(vanilla_scores_file):
        vanilla_ib_scores = load_serialized_object(vanilla_scores_file)
    else:
        print("Computing Vanilla IB Scores")
        vanilla_ib_scores = compute_vanilla_ib_scores(data_loader, data_loader.all_languages, p_m, pu_m)
        vanilla_ib_scores['chain'] = vanilla_ib_scores.apply(lambda x: extract_metadata(x, "chain"), axis=1)
        vanilla_ib_scores['generation'] = vanilla_ib_scores.apply(lambda x: extract_metadata(x, "gen"), axis=1)
        serialize_object(vanilla_ib_scores, vanilla_scores_file)

    systematic_scores_file = data_dir + "systematic_ib_scores.pkl"
    if os.path.exists(systematic_scores_file):
        all_scores = load_serialized_object(systematic_scores_file)
    else:
        print("Computing Systematicity Scores")
        all_attested_scores = systematicity_scoring(data_loader, data_loader.attested_languages, p_m, relations, clusterings, combos_of_interest=combo_queue)
        all_scores = []
        print("Starting combo queue...")
        for combo in tqdm(combo_queue):
            combo_name = "_".join(combo)
            sub_scores = mini_systematicity_scoring(data_loader, combo_name, p_m, relations, clusterings, combo)
            all_scores.append(sub_scores)
        all_scores = pd.concat(all_scores)
        all_scores = pd.concat([all_attested_scores, all_scores])
        all_scores = all_scores.sort_values(by=[ 'language', 'angle_cluster', 'radius_cluster'])
        all_scores = vanilla_ib_scores.join(all_scores)

        all_scores['total_complexity'] = all_scores['complexity'] + all_scores['sys_complexity'] 
        all_scores['total_informativity'] = inf_weight*all_scores['informativity'] +  all_scores['sys_informativity']

        all_scores['total_complexity'] = all_scores['total_complexity'].round(6)
        all_scores['total_informativity'] = all_scores['total_informativity'].round(6)

        all_scores['j'] = all_scores['total_complexity'] - beta*all_scores['sys_informativity']
        serialize_object(all_scores, systematic_scores_file)
            
    return vanilla_ib_scores, all_scores



def add_systematic_hypotheticals(data_loader, combo_queue):

    # Create the (likely) optimal clustering combinations
    # TODO: This should be removed once we fix the simulation code
    print("Adding in Systematic Hypotheticals")
    optimal_partitions = {}
    for combo in combo_queue:
        combo_partition = optimal_partition_for_clusterings((combo[1], combo[0]))
        optimal_partitions["_".join(combo)] =  combo_partition
    optimal_partitions = pd.DataFrame(optimal_partitions).T
    optimal_partitions.columns = data_loader.all_partitions.columns

    # Add the maximally systematic clustering combinations to the data loader
    data_loader.all_partitions = pd.concat([data_loader.all_partitions, optimal_partitions])
    data_loader.hypothetical_languages = optimal_partitions.index.tolist()
    data_loader.all_languages.extend(optimal_partitions.index.tolist())
    data_loader.language_to_lexicon =  data_loader.construct_lexicons()
    data_loader.language_to_encoder = data_loader.construct_encoders()
    return data_loader



def get_optimal_systems_at_zero(scores):
    # This is a TEMPORARY function; will remove once simulated annealing works
    optimal_curve = scores[scores['j'] == 0]
    optimal_curve['beta'] = 2
    optimal_curve = optimal_curve.sort_values(by='total_complexity')
    return optimal_curve



def preprocess_curve(curve):
    curve['gamma'] = 0
    curve = curve.round(10)
    return curve



def analyze_efficiency_over_time(model_name, scores, optimal_curve, fields, labels, opt_params, betas):

    # Compute the efficiency loss for each system

    optimal_curve = preprocess_curve(optimal_curve)

    print(f"Computing optimality for {model_name}")
    optimality_obj = compute_optimality(scores, fields, 
                                        opt_params.tradeoff_signs, 
                                        betas, 
                                        opt_params.gammas,  
                                        opt_params.optimality_dir + model_name, 
                                        theoretical=optimal_curve.drop_duplicates(subset=['beta']))
    
    optimality_info = optimality_obj['attested'].drop_duplicates()
    optimal_scores = scores.iloc[optimality_info['index']]
    optimal_scores = optimal_scores.sort_values(by=['chain', 'generation'])

    # Reformat the scores by chain
    print(f"Computing efficiency by chain for {model_name}")
    efficiency_loss_df = efficiency_by_chain(optimality_info, optimal_scores)

    # Plot deviation from optimality (change to only take 1 argument, no mean)
    image_prefix = f"{opt_params.images_dir}/{model_name}/"
    plot_deviation_from_optimality(efficiency_loss_df, save_name=image_prefix + "deviation_from_optimality.png")

    # Create trajectory plots for each chain
    for chain_id in efficiency_loss_df.index:
        chain_letter = ALPHABET[chain_id]
        chain_data = optimal_scores[optimal_scores['chain'] == chain_id]
        plot_trajectory(chain_data, optimal_curve, fields, labels, save_name = image_prefix + f"chain_{chain_letter}_trajectory.png")

    return efficiency_loss_df






def main(config_file):

    # Set up main variables
    config = load_experiment_config(config_file)

    scores_dir, optimality_dir, images_dir = set_up_directories(config)
    data_loader = load_data_loader(config.domain, config.data_loader_name, config.data_loader_params, load_presaved=config.use_presaved_data_loader)
    p_m = data_loader.load_prior(config.need_param, "array")
    pu_m = data_loader.load_pu_m(config.model_params.model_dump())
    relation_params = config.relation_params
    include_only_convex = config.include_only_convex_clusterings

    relations, clusterings, combo_queue = load_systematicity_information(data_loader, scores_dir, p_m, relation_params, include_only_convex)
    data_loader = add_systematic_hypotheticals(data_loader, combo_queue)

    # Set up constants 
    vanilla_ib_fields = ["complexity", 'informativity']
    vanilla_ib_labels = ["Complexity (bits)", 'Informativity (bits)']

    systematic_ib_fields = ["total_complexity", 'total_informativity']
    systematic_ib_labels = ["Total Complexity (bits)", 'Total Informativity (bits)']

    # Set up optimality parameters
    betas = np.logspace(config.max_beta, 0, 1500)
    optimality_params = OptimalityParams(tradeoff_signs = [1, -1], gammas=[0], 
                                         optimality_dir=optimality_dir, images_dir=images_dir)

    # Score systems
    vanilla_ib_scores, systematic_ib_scores = score_systems(data_loader, scores_dir, p_m, pu_m, 
                                                            relations, clusterings, combo_queue, 
                                                            inf_weight=0, beta=2)
    
    # Extract and visualize the optimal curves
    print("Getting Vanilla IB Theoretical Curve")
    vanilla_curve_file = scores_dir + "vanilla_ib_optimal_curve.pkl"
    if os.path.exists(vanilla_curve_file):
        vanilla_ib_optimal_curve = load_serialized_object(vanilla_curve_file)
    else:
        vanilla_ib_optimal_curve, _ = make_curve(len(data_loader.referents), betas, p_m, pu_m, ctol=3)
        serialize_object(vanilla_ib_optimal_curve, vanilla_curve_file)

    # Extract and visualize the optimal extended curve from the set of systems under consideration (TODO: CHANGE ONCE SIMULATED ANNEALING WORKS)
    print("Getting Systematic IB Theoretical Curve")
    systematic_betas = [2]
    systematic_ib_optimal_curve = get_optimal_systems_at_zero(systematic_ib_scores)

    
    scatter_data = {"All Systems": {"data": vanilla_ib_scores, "marker": None, "colour": None, "size": None}}
    visualize_tradeoff(scatter_data, vanilla_ib_fields, vanilla_ib_labels, optimal_curve=vanilla_ib_optimal_curve,
                       save_name=images_dir + "vanilla_ib/vanilla_ib_curve.png")
    

    attested_scores = systematic_ib_scores[systematic_ib_scores['ground_truth']]
    scatter_data = {"Optimal Systems": {"data": systematic_ib_optimal_curve, "marker": "X", "colour": "red", "size": 50},
                    "All Systems": {"data": attested_scores, "marker": None, "colour": None, "size": None}}
    visualize_tradeoff(scatter_data, systematic_ib_fields, systematic_ib_labels, optimal_curve=systematic_ib_optimal_curve,
                       save_name=images_dir + "systematic_ib/systematic_ib_curve.png")


    # Compute the degree of optimality for each of the systems according to their respective fields (make sure to modify curve variable)
    vanilla_ib_efficiency_loss = analyze_efficiency_over_time("vanilla_ib", vanilla_ib_scores, 
                                                                vanilla_ib_optimal_curve, vanilla_ib_fields, 
                                                                vanilla_ib_labels, optimality_params, betas=betas)
    
    # TODO: REMOVE HEAD 1
    systematic_ib_efficiency_loss = analyze_efficiency_over_time("systematic_ib", systematic_ib_scores,
                                                                  systematic_ib_optimal_curve,
                                                                  systematic_ib_fields, systematic_ib_labels,
                                                                  optimality_params, betas=systematic_betas)


    # Make facet plot of per-system deviation
    visualize_facet_plot(vanilla_ib_efficiency_loss, systematic_ib_efficiency_loss, 
                         save_name=images_dir + "facet_plot.png")
    


if __name__ == "__main__":
    config_file = f"{CONFIG_DIR}/shepard.yaml"

    main(config_file)