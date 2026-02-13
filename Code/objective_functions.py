import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from copy import deepcopy

pd.options.mode.chained_assignment = None



class ObjectiveFunction:

    def __init__(self, fields, signs, save_name):
        self.fields = fields
        self.signs = signs
        self.save_name = save_name


    def compute_objective_function(self, np_df, beta, gamma):
        params = [1, beta, gamma] #1 is for complexity 
        signed_params = [params[i] * self.signs[i] for i in range(len(self.signs))]

        return np.dot(np_df, signed_params)


    def get_optimal_empirical_bound(self, df, betas, gammas):

        optimal_systems_file = self.save_name + "_optimal_systems.csv"

        if os.path.exists(optimal_systems_file):
            optimal_systems = pd.read_csv(optimal_systems_file, index_col=0)
            return optimal_systems
    
        optimal_systems = []
        np_df = df[self.fields].to_numpy()

        for beta in tqdm(betas):
            for gamma in gammas:
                # Get optimal empirical point
                values = self.compute_objective_function(np_df, beta, gamma)
                min_idx = np.argmin(values)
                curr_optimal = df.iloc[[min_idx]]
                curr_optimal[['beta', 'gamma', 'j']] = [beta, gamma, values[min_idx]]
                optimal_systems.append(curr_optimal)

        optimal_systems = pd.concat(optimal_systems).round(10)
        optimal_systems = optimal_systems.sort_values(by=['beta'], ascending=[False])
        optimal_systems.to_csv(optimal_systems_file)
        return optimal_systems


    def find_optimal_tradeoff(self, df, optimal_systems, betas, gammas):

        save_file = self.save_name + "_all_differences.csv"
        if os.path.exists(save_file):
            results = pd.read_csv(save_file, index_col=0)
            return results

        optimal_systems = optimal_systems.set_index(['beta', 'gamma'])

        all_tradeoffs = []
        np_df = df[self.fields].to_numpy()

        for gamma in gammas:
            for beta in tqdm(betas):
                values = self.compute_objective_function(np_df, beta, gamma)
                optimal_value = optimal_systems.loc[beta, gamma]['j']

                curr = pd.DataFrame({"lang_id": df.index, "beta": beta, "gamma": gamma, 
                                     "j": values, "min_at_params": optimal_value, 
                                     "diffs": values - optimal_value})
                
                min_score_per_language = curr.groupby("lang_id")['diffs'].idxmin()
                min_score_df = curr.loc[min_score_per_language]
                all_tradeoffs.append(min_score_df)
        
        all_tradeoffs = pd.concat(all_tradeoffs).reset_index()
        min_score_per_language = all_tradeoffs.groupby("lang_id")['diffs'].idxmin()
        results = all_tradeoffs.loc[min_score_per_language].set_index("lang_id")
        results.to_csv(save_file)
        return results
                

                

def get_optimal_empirical_bound(df, fields, signs, betas, gammas, save_name):
    df = deepcopy(df)

    if isinstance(betas, float):
        betas = np.logspace(betas, 0, num=1500)
    betas = np.round(betas, 10)
    gammas = np.round(gammas, 10)
    
    objective_func = ObjectiveFunction(fields, signs, save_name)
    optimal_systems = objective_func.get_optimal_empirical_bound(df, betas, gammas)
    return optimal_systems



def compute_optimality(df, fields, signs, betas, gammas, save_name, optimal_curve):

    df = deepcopy(df)

    if isinstance(betas, float):
        betas = np.logspace(betas, 0, num=1500)
    betas = np.round(betas, 10)
    gammas = np.round(gammas, 10)
    
    objective_func = ObjectiveFunction(fields, signs, save_name)

    # # Get the optimal theoretical or empirical bound
    # if theoretical is not None:
    #     optimal_systems = deepcopy(theoretical)
    # else:
    #     optimal_systems = objective_func.get_optimal_empirical_bound(df, betas, gammas)

    # Compute the optimality of all systems relative to the bound
    all_systems =  objective_func.find_optimal_tradeoff(df, optimal_curve, betas, gammas)

    # Post-processing
    all_systems = all_systems.loc[df.index]
    all_systems['ground_truth'] = df['ground_truth'].astype(int)
    all_systems['normalized_diffs'] = all_systems['diffs'] / all_systems['beta']
    all_systems = all_systems.round(6)

    attested = all_systems[all_systems['ground_truth'] == 1]
    hypothetical = all_systems[all_systems['ground_truth'] == 0]

    optimality_object = {
        "attested": attested,
        "hypothetical": hypothetical,
        "optimal": optimal_curve,
        "all_systems": all_systems
    }

    return optimality_object