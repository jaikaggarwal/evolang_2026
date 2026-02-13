from utils import *
from partition_manipulation import *
from shepard import *


class DataLoader:

    def __init__(self, domain_name, params):
        """
        domain_name (str): one of kinship, deixis, etc.
        meaning_space List[str]: Ordered meanings (kin types, distance/orientation combinations, etc.)
        """
        self.domain_name =  domain_name
        self.params = params

        self.referents = self.get_domain_referents()
        self.attested_languages, self.hypothetical_languages, self.all_partitions = self.load_partitions()
        self.all_languages = self.attested_languages + self.hypothetical_languages

        self.language_to_lexicon = self.construct_lexicons()
        self.language_to_mapping = self.construct_word_meaning_mappings()
        self.language_to_encoder = self.construct_encoders()


    def load_partitions(self):

        attested_partitions = self.load_attested_partitions()
        attested_partitions = standardize_partitions(attested_partitions)
           
        hypothetical_partitions = self.load_hypothetical_partitions()
        hypothetical_partitions = self.add_extreme_partitions(hypothetical_partitions)
        hypothetical_partitions = standardize_partitions(hypothetical_partitions)
        hypothetical_partitions = remove_duplicate_partitions(attested_partitions, hypothetical_partitions)

        attested_languages = attested_partitions.index.tolist()
        hypothetical_languages = hypothetical_partitions.index.tolist()
        all_partitions = pd.concat([attested_partitions, hypothetical_partitions])

        return attested_languages, hypothetical_languages, all_partitions


    def construct_lexicons(self):
        language_to_lexicon = {}
        for language in tqdm(self.all_partitions.index):
            curr_partition = self.all_partitions.loc[language]
            lexicon_df = convert_partition_to_lexicon(curr_partition, self.referents)
            language_to_lexicon[language] = lexicon_df
        return language_to_lexicon


    def construct_word_meaning_mappings(self):
        language_to_mapping = {}
        for language in self.language_to_lexicon:
            mapping = extract_word_meaning_mapping(self.language_to_lexicon[language])
            language_to_mapping[language] = mapping
        return language_to_mapping


    def construct_encoders(self):
        language_to_encoder = {}
        for language in tqdm(self.language_to_lexicon):
            encoder = convert_lexicon_to_encoder(self.language_to_lexicon[language])
            language_to_encoder[language] = encoder
        return language_to_encoder

    def add_extreme_partitions(self, hypothetical_partitions):
        hypothetical_partitions.loc["one_word"] = np.zeros(len(self.referents))
        hypothetical_partitions.loc['n_words'] = np.arange(len(self.referents))
        return hypothetical_partitions


    def get_domain_referents(self):
        pass


    def load_attested_partitions(self):
        pass


    def load_hypothetical_partitions(self):
        pass

    
    def load_domain_specific_need(self):
        pass


    def load_prior(self, prior_type, data_type):
        if prior_type == "uniform":
            num_referents = len(self.referents)
            prior = np.ones(num_referents) / num_referents
        else:
            prior = self.load_domain_specific_need()
        
        if data_type == "dataframe":
            prior = pd.DataFrame(prior, index=self.referents, columns=["need"])
            
        return prior


    def load_pu_m(self, params):
        pass


    def load_feature_df(self, params):
        pass


    def load_relations(self, params):
        pass


    def visualize_system(self, name):
        pass

    def visualize_partition(self, partition):
        pass

    

class ShepardDataLoader(DataLoader):

    def __init__(self, params):
        domain_name = "shepard"
        super().__init__(domain_name, params)
    
    
    def get_domain_referents(self):
        return SHEPARD_REFERENTS
    

    def load_attested_partitions(self):
        return load_attested_shepard_partitions()


    def load_hypothetical_partitions(self):
        return load_hypothetical_shepard_partitions()
    
    
    def load_domain_specific_need(self):
        return load_shepard_prior()


    def load_pu_m(self, params):
        return load_shepard_pu_m(params)

    
    def load_feature_df(self, params):
        return load_shepard_feature_df()
    

    def load_relations(self, params):
        relations = load_shepard_relations(params.model_params)
        return relations


    def visualize_system(self, name):
        curr_partition = self.all_partitions.loc[name]
        visualize_shepard_system(curr_partition, name)


    def visualize_partition(self, partition):
        visualize_shepard_system(partition, "")
    


def load_data_loader(domain, name, params, load_presaved):

    file_path = f"{DATA_LOADER_DIR}/{domain}_{name}.pkl"

    if load_presaved:
        data_loader = load_serialized_object(file_path)
        return data_loader

    if domain == "shepard":
        data_loader = ShepardDataLoader(params)
    else:
        raise Exception("Please select one of the following domains: shepard.")
    
    serialize_object(data_loader, file_path)
    return data_loader