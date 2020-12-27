import collections
import fasttext
import gzip
import json
import math
import matplotlib
import nltk
import os
import pickle
import random
import scipy.sparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zstandard as zstd

from collections import Counter
from gensim.models.coherencemodel import CoherenceModel
from joblib import dump, load
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import LongType, StructField, StructType
from pyspark.ml.clustering import LDA, LDAModel, LocalLDAModel
from pyspark.ml.linalg import Vectors, SparseVector
from scipy.sparse import dok_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from utils import *


def create_spark_session(n_jobs, executor_mem, driver_mem):
    '''
    Initialize a spark SparkSession

    Parameters
    ----------
    n_jobs : int
        The number of jobs used in the spark session. '-1' means using all the processors
    executor_mem : int
        The memory in g for each executor
    driver_mem : int
        The memory in g for the driver

    Returns
    ----------
    spark : SparkSession
        A Sparsession according to the given parameters
    conf : SparkConf
    '''

    n_jobs_local = ''
    if n_jobs == 1:
        n_jobs_local = 'local'
    elif n_jobs == -1:
        n_jobs_local = 'local[*]'
    else:
        n_jobs_local = 'local[' + str(n_jobs) + ']'

    conf = SparkConf().setMaster(n_jobs_local).setAll(
        [('spark.executor.memory', str(executor_mem) + 'g'), ('spark.driver.memory', str(driver_mem) + 'g')])

    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    return spark, conf


def get_relevant_channels(min_n_sub):
    '''
    Paremeter
    ----------
    min_n_sub : int
        The minimum number of subscribers for a channel to be relevant

    Returns
    ----------
    Set of channel_id, which correspond to all the relevant channel
    '''
    df_channelcrawler = pd.read_csv(
        '/dlabdata1/youtube_large/df_channels_en.tsv.gz', sep='\t')

    # Filter channels with at least min_n_sub subs
    df_channelcrawler = df_channelcrawler[df_channelcrawler['subscribers_cc'] >= min_n_sub]

    return set(df_channelcrawler['channel'])


def check_10000_views(video):
    '''
    Parameter
    ----------
    video :

    Returns
    ----------
    Boolean which is True if the video has more than 10'000 views
    '''
    try:
        view_counts = video['view_count']
        if view_counts != None:
            return view_counts >= 10000
        else:
            return False
    except KeyError:
        return False


def check_channel(video, set_relevant_channels):
    '''
    Parameters
    ----------
    video:
    set_relevant_channels : set
        Set of all the channels with more than 100'000 subscribers

    Returns
    ----------
    Boolean which is True if the video comes from a channel with more than 100'000 subscrbers

    '''
    try:
        return video['channel_id'] in set_relevant_channels
    except:
        return False


def isEnglishAlpha(s):
    '''
    Parameter
    ----------
    s : string

    Returns
    ----------
    Boolean which is True if s is a string from the english alphabet
    '''
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_freq_tokens_per_video(video, use_bigram=False):
    '''
    Parameters
    ----------
    video :
    use_bigram : boolean
        if True, generate bi_grams

    Returns
    ----------
    Dictionnary with tokens as key and the number of occurence as value
    '''

    title_tokens = [w for w in tokenizer.tokenize(
        video['title'].lower()) if not w in stop_words]
    tag_tokens = [w for w in tokenizer.tokenize(
        video['tags'].lower()) if not w in stop_words]

    # We want to keep duplicates !!
    tokens_per_video = title_tokens + tag_tokens

    # Filter token with length < 3, with non english alphabet since fastext is not 100% accurate and remove numerical token
    tokens_keep = []
    for token in tokens_per_video:
        if len(token) >= 3 and (not token.isnumeric()) and isEnglishAlpha(token):
            tokens_keep.append(token)

    # Stemming
    stemmed_tokens_per_video = [s_stemmer.stem(w) for w in tokens_keep]

    # Generate bigrams
    if use_bigram:
        stemmed_tokens_per_video.extend(
            nltk.bigrams(set(stemmed_tokens_per_video)))

    # Return a Counter object of the tokens
    return collections.Counter(stemmed_tokens_per_video)


def remove_zero_rows(M):
    '''Function that removes all rows from sparse matrix M that contains only zero.

    Parameter
    ----------
    M : sparse matrix

    Returns
    ----------
    The same sparse matrix as in parameter without the zero rows
    '''
    num_nonzeros = np.diff(M.indptr)
    return M[num_nonzeros != 0]


def fill_underlying_dict(freq_tokens_per_video, word2id, i_vid):
    '''Method to fill the underlying dictionnary in order to
    update the sparse matrix incrementally by videos

    Parameters
    ----------
    freq_tokens_per_video : dict
        Dictionnary with tokens as key and the number of occurence as value

    word2id : dict
        Dictionnary with token as key and its index (id_token) as value

    i_vid : int
        Index of the video from the dataset

    Returns
    ----------
    dict_freq_tokens_for_sparse_matrix : dict
        Dictionnary of (id_vid, id_token) as key and the number of occurence
        of the token as value.
    '''

    dict_freq_tokens_for_sparse_matrix = {}

    for key in freq_tokens_per_video.keys():

        # Column index in the sparse matrix (one column for each token)
        try:
            j_token = word2id[key]

            # Filling the underlying dict
            dict_freq_tokens_for_sparse_matrix[(
                i_vid % 1000000,   j_token)] = freq_tokens_per_video[key]

        except KeyError:
            None

    return dict_freq_tokens_for_sparse_matrix


def get_word2id(set_stemmed_tokens):
    '''
    Create dictionnary of tokens with their indice

    Parameter
    ----------
    set_stemmed_tokens : set
        Set of the vocabulary from the dataset

    Returns
    ----------
    word2id : dict
        Dictionnary with token as key and its index (id_token) as value
    id2word : dict
        Dictionnary with id_token as key and the corresponding token as value
    '''
    word2id = {}

    # Fill dictionnary of tokens
    for i, token in enumerate(set_stemmed_tokens):
        word2id[token] = i

    id2word = {v: k for k, v in word2id.items()}

    return word2id, id2word


def save_to_pickle(var, filename, path_write_data):
    '''
    Parameters
    ----------
    var : any python variable
        variable to be saved
    filename : string
        Name of the pickle we will create
    path_write_data : string
        Path to the folder where we write and keep intermediate results
    '''
    with open(path_write_data + filename + '.pickle', 'wb') as f:
        f.dump(var)
    f.close()


def get_vocab(path_dataset, set_relevant_channels):
    '''
    First pass on the whole dataset in order to construct the vocabulary and to
    keep the indexes of all the relevant videos for topic modelling.

    Parameters
    -----------
    path_dataset : string
        Path to the folder that contains all files
    set_relevant_channels : setAll
        Set of channel_id, which correspond to all the relevant channel

    Returns
    -----------
    set_stemmed_tokens : set
        Set of the vocabulary from the dataset
    set_relevant_vid : set
        Set of the indexes of all relevant video
    '''
    # Variable that contains the idx of every non english vid and that
    # belongs to a channel in channelcrawler.csv TO BE USED IN SECOND ITER
    set_relevant_vid = set()

    # Variable first instanciated as set to check existing tokens efficiently,
    # which will be a list in order to get the index for each tokens
    set_stemmed_tokens = set()

    # Reading the file
    with gzip.open(path_dataset, 'rb') as f:

        for i, line in enumerate(f):

            # line is a byte dict, video is the corresponding dict
            video = json.loads(line)

            if check_channel(video, set_relevant_channels) and check_10000_views(video):

                tokens_per_video = get_freq_tokens_per_video(
                    video, use_bigram)

                set_relevant_vid.add(i)
                set_stemmed_tokens.update(tokens_per_video)

            if i % 10000000 == 0 and i != 0:
                print('Processed ' + str(i) + ' videos...')

    return set_stemmed_tokens, set_relevant_vid


def get_document_term_matrix(path_dataset, path_write_data, set_relevant_vid, word2id, use_bigram):
    '''
    Parameters
    ----------
    path_dataset : string
        Path to the folder that contains all files
    path_write_data : string
        Path to the folder where we write and keep intermediate results
    set_relevant_vid : set
        Set of the indexes of all relevant video
    word2id : dict
        Dictionnary with token as key and its index (id_token) as value

    Returns
    ----------
    S : sparse matrix
        The document-term matrix from the data
    '''

    i_vid = 0
    n_mat_created = 0

    # Reading the file
    with gzip.open(path_dataset, 'rb') as f:

        for i, line in enumerate(f):

            if i_vid % 1000000 == 0 and i_vid != 0:

                file_name = 'S' + str(int(i_vid / 1000000)) + '.npz'

                if not os.path.isfile(path_write_data + 'matrices/' + file_name):

                    # Transform to csr format for memory efficiency
                    S = S.tocsr()
                    scipy.sparse.save_npz(
                        path_write_data + 'matrices/' + file_name, S)
                    n_mat_created += 1

                    # Refresh mini sparse matrix
                    S = dok_matrix(
                        (1000000, size_of_tokens_dict), dtype=np.uint8)

            if i in set_relevant_vid:

                video = json.loads(line)

                # Get the tokens for each video and theirs number of occurences
                freq_tokens_per_video = get_freq_tokens_per_video(
                    video, use_bigram)

                # Fill the underlying dict
                dict_freq_tokens_for_sparse_matrix = fill_underlying_dict(
                    freq_tokens_per_video, word2id, i_vid)

                # Fill data in to sparse matrix
                dict.update(S, dict_freq_tokens_for_sparse_matrix)

                # Increase i_vid
                i_vid += 1

            if i % 10000000 == 0 and i != 0:
                print('Processed ' + str(i) + ' videos...')

            i += 1

    # Save last sparse matrix
    S = S.tocsr()
    scipy.sparse.save_npz(path_write_data + 'matrices/S_last.npz', S)

    # Get full sparse matrix by stacking all intermediate matrices
    S = scipy.sparse.load_npz(path_write_data + 'matrices/S1.npz')

    for i in range(2, n_mat_created):
        S_next = scipy.sparse.load_npz(
            path_write_data + 'matrices/S' + str(i) + '.npz')
        S = scipy.sparse.vstack([S, S_next])

    # Add last matrix
    S_last = scipy.sparse.load_npz(path_write_data + 'matrices/S_last.npz')
    S = scipy.sparse.vstack([S, S_last])

    return S


def get_vid_for_tm(set_relevant_vid, n_top=20):
    '''
    Select only the 'n_top' videos with the most views for each combination of
    'category', 'uploaded_year' and 'channel_id'. These videos will be used for
    training the topic model.

    Parameters
    ----------
    set_relevant_vid : set
        Set of the indexes of all relevant video
    n_top : int

    Returns
    ----------
    set_relevant_vid_top20 : set
    '''
    # create pandas DataFrame of relevant videos with relevant features
    columns_names = ['channel_id', 'view_counts', 'uploaded_year', 'category']

    # store relevant features of relevant videos in a list
    list_relevant_data = []

    # Read file
    with gzip.open('/dlabdata1/youtube_large/yt_metadata_en.jsonl.gz', 'rb') as f:

        for i, line in enumerate(f):

            if i in set_relevant_vid:

                # line is a str dict, video is the dict corresponding to the str dict
                video = json.loads(line)

                list_vid_relevant_features = [video['channel_id']]
                list_vid_relevant_features.append(video['view_count'])
                list_vid_relevant_features.append(video['upload_date'][:4])
                list_vid_relevant_features.append(video['categories'])

                list_relevant_data.append(list_vid_relevant_features)

    # Get DataFrame
    df_relevant_data = pd.DataFrame(list_relevant_data, columns=columns_names)

    # Keep the n_top vid for each combination of 'category', 'uploaded_year' and 'channel_id'
    df_relevant_data_top20 = df_relevant_data.sort_values(['view_counts'], ascending=False).groupby(
        ['category', 'uploaded_year', 'channel_id']).head(n_top)

    # Create set of index of the selected videos
    set_relevant_vid_top20 = set(sorted(df_relevant_data_top20.index.values))

    return set_relevant_vid_top20


def get_document_term_matrix_for_tm(S, set_relevant_vid_final, id2word, min_vid_per_token):
    '''
    Parameters
    ----------
    S : sparse matrix
        The document-term matrix from the data
    set_relevant_vid_final : set
        Set of indexes of videos used for topic modelling
    id2word : dict
        Dictionnary with id_token as key and the corresponding token as value
    min_vid_per_token : int
        Threshold for tokens that appear in at least min_vid_per_token videos
        to be considered as relevant tokens.

    Returns
    ----------
    S_final : sparse matrix
        The document-term matrix of the data used for topic modelling
    id2word_final : dict
        Dictionnary with id_token as key and the corresponding token as value.
        Keep only the filtrered tokens
    '''
    S_final = S[set_relevant_vid_top20]

    # Convert matrix to csc for efficient computing
    S_final = S_final.tocsc()

    list_relevant_tokens = []

    # Iterate on the columns
    for i in range(S_final.shape[1]):
        # Check column has more than 100 non zero entries
        if S_final[:, i].count_nonzero() >= min_vid_per_token:
            list_relevant_tokens.append(i)

    S_final = S[:, list_relevant_tokens].tocsr()
    S_final = remove_zero_rows(S_final)

    # Construct dict of token_id to tokens
    id2word_final = {}

    for i, id_token in enumerate(list_relevant_tokens):
        id2word_final[i] = id2word[id_token]

    return S_final, id2word_final


def get_dict_for_row(S, row):
    '''
    Construct SparseVector bag-of-word for each row (videos)

    Parameters
    ----------
    S : sparse matrix
    row : iterable

    Returns
    ----------
    A SparseVector
    '''
    tmp_dict = {}
    for key, value in row:
        tmp_dict[key[1]] = value

    return SparseVector(S.shape[1], tmp_dict)


def get_data_for_tm(S_final, spark):
    '''
    Parameters
    ----------
    S_final : sparse matrix
        The document-term matrix of the data used for topic modelling
    spark : SparkSession

    Returns
    ----------
    data_df : Spark DataFrame
        A Spark DataFrame of the data used for topic modelling
    '''
    data = []

    for i in range(S_final.shape[0]):
        data.append([i, get_dict_for_row(
            S_final.getrow(i).todok().items(), S_final)])

    # Construct dataframe for LDA
    data_df = spark.createDataFrame(data, ["id", "features"])

    return data_df
