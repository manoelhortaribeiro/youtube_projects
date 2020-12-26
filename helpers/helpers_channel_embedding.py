import pickle
import os

import numpy as np
import pandas as pd

from annoy import AnnoyIndex
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

'''
Retrieve the array obtained by apllying the dimentionality reduction algorithm
graph_matrix: SHAPE: (channels, n_comp)

PARAMETER:
    - file_path: the path where the embedding graph is stored

RETURN: 
    - df: DataFrame representing the graph in the embedding space
'''
def create_dataframe_in_embedding_space_pytorch(model_path):
    graph_matrix = torch.load(model_path)['embedding'].cpu().detach().numpy()
    df = pd.DataFrame(graph_matrix)
    df = df.rename(lambda x: 'dr'+str(x), axis='columns')
    return df

def get_dataframe_in_embedding_space(model_path, embedding_type = 'pytorch'):
    if embedding_type == 'pytorch':                          
        return pd.read_csv(model_path, compression='gzip')
    elif embedding_type == 'word2vecf':
        graph_matrix = np.load(model_path)
        df = pd.DataFrame(graph_matrix)
        df = df.rename(lambda x: 'dr'+str(x), axis='columns')
        df['index'] = np.load(model_path.split('.')[0] + 'index_ordering.npy').astype(int)
        df = df.sort_values(by=['index'])
        return df.set_index('index')
    else:
        graph_matrix = np.load(file_path)
        graph_matrix = graph_matrix['arr_0']
        df = pd.DataFrame(graph_matrix)
        df = df.rename(lambda x: 'dr'+str(x), axis='columns')
        return df
        
    


'''
Retrieve the array obtained by apllying the dimentinality reductin algorithm
graph_matrix: SHAPE: (channels, n_comp)

PARAMETERS:
    - df_embedding: DataFrame representing the graph in the embedding space
    - n_comp: number of components to use after the dimentionalit reduction

RETURN: The annoy index
'''
def get_annoy_index(df):
    index = AnnoyIndex(df.shape[1], "euclidean")  # Length of item vector that will be indexed
    df.apply(lambda row: index.add_item(row.name, np.array(row)), axis = 1)
    index.build(100) # 100 trees
    return index


def get_k_nearest_neighbors(path, ref_index_channel, dict_ind_channel, k = 20, embedding_type = 'pytorch'):
    df = get_dataframe_in_embedding_space(path, embedding_type)
    index = get_annoy_index(df)
    nearest_neighbors_index = index.get_nns_by_item(ref_index_channel, k)
    nearest_neighbors_id = [dict_ind_channel[val] for val in nearest_neighbors_index]
    return nearest_neighbors_id



def get_random_walk(df_embedding):
    with open("/dlabdata1/youtube_large/jouven/channels_more_300/channels_tuple_random_walk.pkl",'rb') as f:
         random_walk_channels = pickle.load(f)
    f.close()
    random_walk_distance = 0
    for val in random_walk_channels:
        random_walk_distance += distance.euclidean(df_embedding.iloc[val[0]], df_embedding.iloc[val[1]])
    return random_walk_distance

def get_random_walk_new(df_embedding):
    with open("/dlabdata1/youtube_large/jouven/channels_more_300/channels_tuple_random_walk_modified.pkl",'rb') as f:
         random_walk_channels = pickle.load(f)
    f.close()
    random_walk_distance = 0
    for val in random_walk_channels:
        random_walk_distance += distance.euclidean(df_embedding.iloc[val[0]], df_embedding.iloc[val[1]])
    return random_walk_distance



'''
Get the position of ref_channel relative to second_channel in terms of its nearest neighbors ranking.

PARAMETER:
    - ref_channel: The reference channel on which wwe compute it's k nearest neighbor
    - second_channel: The channel where we compute it's ranking relatively to ref_channel
    - dist: Euclidean distance between ref_channel and second_channel
    - index: annoy index
    - df_embedding: DataFrame representing the embedding space

RETURN: The position of second_channel relatively to ref_channel in terms of it's ranking

'''
def get_ranking_position_between_channels(ref_channel, second_channel, index, df_embedding):
    
    nearest_neighbors_index = index.get_nns_by_item(ref_channel, len(df_embedding), search_k = 100000000)
    dist_k_th_nearest = distance.euclidean(df_embedding.iloc[ref_channel], 
                                           df_embedding.iloc[nearest_neighbors_index[len(nearest_neighbors_index)-1]])
    for i in range(0, len(nearest_neighbors_index)):
        if nearest_neighbors_index[i] == second_channel:
            return i
        
        
def get_user_walk_and_position_ratio(files, channels_tuple, embedding_type = 'pytorch'):

    users_walk_tab = []
    users_walk_tab_new = []
    ranking_position_tab = []

    len_random_set = len(channels_tuple)

    for file in files: 
        print('file ', file)                  
        df_embedding = get_dataframe_in_embedding_space(file, embedding_type)     
        n_comp = df_embedding.shape[1]
        print('n_comp ', n_comp)
        random_walk_distance = get_random_walk(df_embedding)
        random_walk_distance_new = get_random_walk_new(df_embedding)
        index = get_annoy_index(df_embedding)
        users_walk = 0
        ranking_position = 0

        for ref_channel, second_channel in channels_tuple:
            users_walk += distance.euclidean(df_embedding.iloc[ref_channel], df_embedding.iloc[second_channel])
            ranking_position += get_ranking_position_between_channels(ref_channel, second_channel, index, df_embedding)

        users_walk_tab.append(users_walk/random_walk_distance)
        users_walk_tab_new.append(users_walk/random_walk_distance_new)
        ranking_position_tab.append(ranking_position/(len_random_set*df_embedding.shape[0]))
        
        return users_walk_tab, users_walk_tab_new, ranking_position_tab
    
    
    
####### Helpers for axis projection 


def get_dataframe_in_embedding_space_limited_channels(file_path, selected_channels):
    df = pd.read_csv(file_path, compression='gzip').reset_index()
    df = df.loc[df['index'].isin(selected_channels)]
    return df.drop(columns=['index'])
    
'''
For the given ref_channel, compute it's k neirest neighbor and create pairs of channels between the found channel and ref_channel

PARAMETERS:
    - channels_pairs: table representing the pair of channel already computed
    - ref_channel: the channel on which we compute the neirest neighbor search
    - index: the annoy index to do the k nearest neighbor search
    - k: the number of neighbors 

'''
def create_pairs(channels_pairs, ref_channel, index, k):
    nearest_neighbors = index.get_nns_by_item(ref_channel, k)
    for neighbor_channel in nearest_neighbors:
        channels_pairs.append((ref_channel, neighbor_channel))
        
        
'''
Generate the set of all pairs of channels with their k neirest neighbors

PARAMETERS:
    - df_embedding: DataFrame representing the channel embedding
    - k: the parameter of the nearest neighbor search
    - n_comp: the number of components after applying the dimensionality reduction
RETURN:
    - list of channels tuple 
'''
def channels_with_neighbors_pairs(df_embedding, k, n_comp, seed):
    channels_pairs = []
    index = get_annoy_index(df_embedding)
    
    for channel in df_embedding.index:
        if not channel == seed[0] and not channel == seed[1]:
            create_pairs(channels_pairs, channel, index, k)
    return pd.DataFrame(channels_pairs)


'''
Creates the axis vector representing the desired cultural concept which is based on the seed pair.

PARAMETERS:
    - path: the path where the reducted matrix is saved
    - k: the number of neirest neighbor
    - seed: the seed pair representing the base of the axis
    - nb_selected_pairs: number of selected pairs to create the axis
RETURN:
    - All the channels pairs ranked by the cosine similarity metric (from higher to lower)
'''

def compute_axis_vector_based_on_seed(path, k, seed, nb_selected_pairs, selected_channels, channel_crawler, dict_ind_channel):
    
    # DataFrame representing the embedding
    df_embedding = get_dataframe_in_embedding_space_limited_channels(path, selected_channels)
    
    n_comp = df_embedding.shape[1]

    channels_pairs = channels_with_neighbors_pairs(df_embedding, k, n_comp, seed)
    
    # Vectors representing the difference between every pairs of channels in channels_pairs
    vector_diff_channels_pairs = np.array(channels_pairs.apply(lambda row: df_embedding.loc[row[0]] - df_embedding.loc[row[1]], axis = 1))
    
    # Vector difference between the seed pair
    vector_diff_seed = np.array(df_embedding.loc[seed[0]] - df_embedding.loc[seed[1]])
    
    # compute cosine similarity score
    similarity_score = cosine_similarity(vector_diff_channels_pairs, vector_diff_seed.reshape(1, -1))
    channels_pairs['similarity'] = similarity_score
    channels_pairs = channels_pairs.sort_values(by = ['similarity'], ascending = False)
    
    # Sort the similarity_score in decreasing order
    #dict_channel_similarity = {}
    #for ind in range(len(channels_pairs)):
    #    dict_channel_similarity[channels_pairs[ind]] = similarity_score[ind]
    #sorted_similarity_score = sorted(dict_channel_similarity.keys(), key=dict_channel_similarity.get, reverse = True)
    
    return cultural_concept_vector(df_embedding, channels_pairs, vector_diff_seed, nb_selected_pairs, seed, channel_crawler, dict_ind_channel)


'''
The nb_selected_pairs-1 pairs are selected based on the cosine similarity score to end up with nb_pairs_selected 
pairs to create the axis (with the original seed pair).
To create the axis, the vector difference of all nb_pairs_selected are averaged together to obtain a single vector 
for the axis that robustly represents the desired cultural concept.

PARAMETERS:
    - df_embedding: DataFrame representing the channel embedding
    - sorted_similarity_score: list of channel pairs ordered by their cosine similarity score
    - vector_diff_seed: vector difference between the seed pair
    - nb_selected_pairs: number of selected pairs to create the axis
RETURN:
    - Vector for the axis that represents the desired cultural concept
'''

def cultural_concept_vector(df_embedding, sorted_similarity_score, vector_diff_seed, nb_selected_pairs, seed, channel_crawler, dict_ind_channel):
    
    df_output = pd.DataFrame() # Dataframe to print the selected pairs
    
    selected_channels_pairs = [seed] # All the selected channels pairs
    
    # Select and print the channels that we take to create the ais
    count_selected_pairs = 0 # Counter to keep track of how many pairs we have selected so far
    idx = 0 # Indice counter
    channels_already_taken = [seed[0], seed[1]]
    while count_selected_pairs <= nb_selected_pairs:
        pair = sorted_similarity_score.iloc[idx]
        
        # We don't want channels to be selected multiple times
        if not(pair[0] in channels_already_taken or pair[1] in channels_already_taken):
            df_output = df_output.append(channel_crawler[channel_crawler['channel'] == dict_ind_channel[pair[0]]])
            df_output = df_output.append(channel_crawler[channel_crawler['channel'] == dict_ind_channel[pair[1]]])
            
            selected_channels_pairs.append((pair[0], pair[1]))
            
            channels_already_taken.append(pair[0])
            channels_already_taken.append(pair[1])
            count_selected_pairs += 1
        idx += 1
    with open("/dlabdata1/youtube_large/jouven/channels_more_300/embedding_pairs.pkl",'wb') as f:
        extended = pickle.dump(channels_already_taken, f)
    f.close()
    cultural_concept_vectors = [vector_diff_seed] # Vectors of the difference of the selected pairs
    # Create the axis vector by taking the mean of the vector difference of the selected pairs
    for channel_pair in selected_channels_pairs:
        cultural_concept_vectors.append(np.array(df_embedding.loc[channel_pair[0]]) - np.array(df_embedding.loc[channel_pair[1]]))
    cultural_concept_vectors = np.array(cultural_concept_vectors)
    
    return cultural_concept_vectors.mean(axis = 0), df_output

##### Useful functions for plotting the embedding

def channel_to_name(channelcrawler, dict_channel_ind):
    dict_channel_name = {}
    
    channelcrawler['channel_idx'] = channelcrawler['channel'].apply(lambda x: dict_channel_ind[x])
                                                   
    dict_idx_name = channelcrawler[['channel_idx', 'name_cc']].set_index('channel_idx').to_dict()['name_cc']
    dict_name_idx = channelcrawler[['name_cc', 'channel_idx']].set_index('name_cc').to_dict()['channel_idx']
    
    return dict_idx_name, dict_name_idx



    