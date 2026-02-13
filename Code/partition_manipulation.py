
from utils import *
from information_theory import *
from multiprocessing import Pool



def convert_partition_to_lexicon(row, meaning_space):
    lexicon_df = pd.DataFrame(row).reset_index()
    lexicon_df.columns = ['parameter_id', 'word_id']
    lexicon_df = lexicon_df.set_index('parameter_id')
    lexicon_df = lexicon_df.loc[meaning_space].reset_index()
    return lexicon_df.dropna()



def convert_lexicon_to_encoder(df):

    # Required to help pivot the table
    df['tmp_var'] = 1
    encoder_df = df.pivot(index="parameter_id", columns="word_id", values="tmp_var")

    referents = df['parameter_id'].drop_duplicates()
    encoder_df = encoder_df.loc[referents]
    encoder_df = encoder_df.fillna(0)
    encoder = encoder_df.to_numpy()

    encoder = encoder/encoder.sum(axis=1).reshape((-1, 1))

    # The number of 1s present should be the same as the number of parameters being considered
    # Normalization is not required because each referent is named uniquely
    assert np.sum(encoder) == len(referents)
    return encoder 



def convert_encoder_to_partition(encoder):
    # Convert encoder to partition
    partition = np.argmax(encoder, axis=1)
    
    # Standardize the labels
    partition = canonical_label(partition)
    return partition



def filter_unique_partitions(partitions):

    unique_partitions = []
    unique_indices = []
    seen_partitions = []

    for i, partition in enumerate(partitions):
        code = "_".join(map(str, partition))
        if code in set(seen_partitions):
            continue
        
        unique_partitions.append(partition)
        unique_indices.append(i)
        seen_partitions.append(code)

    return unique_partitions, unique_indices


def invert_word_meaning_mapping(curr_mapping):
    meaning_to_word_mapping = {}
    for word in curr_mapping:
        meanings = curr_mapping[word]
        for meaning in meanings:
            meaning_to_word_mapping[meaning] = word
    return meaning_to_word_mapping



def extract_word_meaning_mapping(lexicon):
    return lexicon.groupby("word_id")['parameter_id'].agg(list).to_dict()



def find_referent_by_conditions(df, *conditions):
    df = deepcopy(df)
    for condition in conditions:
        feature, val = condition
        df = df[df[feature] == val]
    return df



def canonical_label(partition):

    if type(partition) == list:
        partition = np.array(partition).reshape(1, -1)
    elif type(partition) == pd.Series:
        partition = np.array(partition).reshape(1, -1)
    elif (type(partition) == np.ndarray) and (partition.ndim == 1):
        partition = np.array(partition).reshape(1, -1)

    num_objects = partition.shape[1]

    sorted_vals, first_indices, all_indices = np.unique(partition, return_index=True, return_inverse=True) 
    sorted_first_indices = np.argsort(np.argsort(first_indices))
    new_arr = np.arange(num_objects)
    return new_arr[sorted_first_indices][all_indices]



def standardize_partitions(all_partitions):
    standardized_partitions = []
    for i, row in all_partitions.iterrows():
        canonical_partition = canonical_label(row)
        standardized_partitions.append(canonical_partition)
    standardized_partitions = pd.DataFrame(standardized_partitions, columns=all_partitions.columns, index=all_partitions.index)
    return standardized_partitions



def remove_duplicate_partitions(base_df, alt_df):
    base_codes = base_df.astype(str).sum(axis=1)
    alt_codes = alt_df.astype(str).sum(axis=1)

    no_duplicates_df = alt_df[~alt_codes.isin(base_codes)]
    return no_duplicates_df.drop_duplicates()