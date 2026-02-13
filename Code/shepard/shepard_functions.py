
import numpy as np
import pandas as pd
from shepard.shepard_constants import SHEPARD_REFERENTS, SHEPARD_ANGLES, SHEPARD_RADIUSES
from shepard.helper_functions.tools import read_json_file
import os
# ----------------- Loading Partition Data ----------------- #

def load_attested_shepard_partitions():

    data_dir = os.path.dirname(__file__)

    data = read_json_file(f"{data_dir}/helper_functions/exp2_chains.json")
    chains = data['chains']

    system_to_partition = {}
    for chain in chains:
        generations = chain['generations']
        for generation in generations:
            system_id = f"chain_{chain['chain_id']}_gen_{generation['generation_number']}"
            productions = generation['productions']
            system_to_partition[system_id] = productions

    partitions_df = pd.DataFrame(system_to_partition).T
    partitions_df.columns = SHEPARD_REFERENTS
    return partitions_df


def load_hypothetical_shepard_partitions():
    empty_df = pd.DataFrame(columns=SHEPARD_REFERENTS)
    return empty_df


# ----------------- Loading Feature Dataframe ----------------- #
def load_shepard_feature_df():
    angles = np.repeat(SHEPARD_ANGLES, len(SHEPARD_RADIUSES))
    radiuses = np.tile(SHEPARD_RADIUSES, len(SHEPARD_ANGLES))
    return pd.DataFrame({"angle": angles, "radius": radiuses}, index=SHEPARD_REFERENTS)



# ----------------- Prior ----------------- #
def load_shepard_prior():
    num_referents = len(SHEPARD_REFERENTS)
    prior = np.ones(num_referents) / num_referents
    return prior



# ----------------- Semantic Space ----------------- #

def load_shepard_pu_m(params):
    alpha = params['weights'][0]
    gamma = params['mu']
    referents = [(a, r) for a in range(len(SHEPARD_ANGLES)) for r in range(len(SHEPARD_RADIUSES))]
    num_referents = len(referents)
    m_t_u = np.zeros((num_referents, num_referents))

    for t in range(num_referents):
        t_vec = referents[t]
        sims = []
        for u in range(num_referents):
            u_vec = referents[u]
            d_tu = alpha * ((u_vec[0] - t_vec[0])**2) + (1-alpha) * ((u_vec[1] - t_vec[1])**2)
            # sims.append(d_tu) # sanity check
            sims.append(np.exp(-gamma * d_tu))
        sims = np.array(sims)
        m_t_u[t, :] = sims
    m_t_u = m_t_u / np.sum(m_t_u, axis=1)[:, np.newaxis]
    return m_t_u



# ----------------- Systematicity  ----------------- #

# def load_shepard_relations(feature_df):
#     # Need to map each meaning to the correct feature dimension
#     # In theory, if the feature_df is structured as two ordinal columns, as in deixis, we can do the same thing as deixis
#     meanings = feature_df.index

#     angle_relation = pd.DataFrame(feature_df['angle'])
#     angle_relation['tmp_var'] = 1
#     angle_relation = angle_relation.pivot(columns="angle", values="tmp_var").fillna(0)
#     angle_relation = angle_relation.loc[meanings]

#     radius_relation = pd.DataFrame(feature_df['radius'])
#     radius_relation['tmp_var'] = 1
#     radius_relation = radius_relation.pivot(columns="radius", values="tmp_var").fillna(0)
#     radius_relation = radius_relation.loc[meanings]

#     relations = {"angle": angle_relation, "radius": radius_relation}
#     return relations


def load_shepard_relations(params):
    gamma = params['mu']
    referents = [(a, r) for a in range(len(SHEPARD_ANGLES)) for r in range(len(SHEPARD_RADIUSES))]
    num_referents = len(referents)
    m_t_u = np.zeros((num_referents, len(SHEPARD_ANGLES)))

    for t in range(num_referents):
        t_vec = referents[t]
        sims = []
        for u in range(len(SHEPARD_ANGLES)):
            d_tu = ((u - t_vec[0])**2)
            sims.append(np.exp(-gamma * d_tu))
        sims = np.array(sims)
        m_t_u[t, :] = sims
    m_t_u = m_t_u / np.sum(m_t_u, axis=1)[:, np.newaxis]
    angle = pd.DataFrame(m_t_u, index=SHEPARD_REFERENTS, columns=SHEPARD_ANGLES)


    m_t_u = np.zeros((num_referents, len(SHEPARD_RADIUSES)))

    for t in range(num_referents):
        t_vec = referents[t]
        sims = []
        for u in range(len(SHEPARD_RADIUSES)):
            d_tu = ((u - t_vec[1])**2)
            sims.append(np.exp(-gamma * d_tu))
        sims = np.array(sims)
        m_t_u[t, :] = sims
    m_t_u = m_t_u / np.sum(m_t_u, axis=1)[:, np.newaxis]
    radius = pd.DataFrame(m_t_u, index=SHEPARD_REFERENTS, columns=SHEPARD_RADIUSES)
    return {"angle": angle, "radius": radius}


