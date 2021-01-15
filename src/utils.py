import collections
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


class FakedGensimDict:
    """
    Locally made class for `~gensim.corpora.dictionary.Dictionary`
    """

    def __init__(self, data, S):
        if not isinstance(data, dict):
            raise ValueError('`data` must be an instance of `dict`')

        self.id2token = data
        self.token2id = {v: k for k, v in data.items()}
        self.doc2bow = S

    @staticmethod
    def from_vocab(vocab):
        return FakedGensimDict(dict(zip(range(len(vocab)), vocab)))


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
        [('spark.executor.memory', str(executor_mem) + 'g'), ('spark.driver.memory', str(driver_mem) + 'g'), ('spark.driver.maxResultSize', '0')])

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


def check_views(video, n_min_views):
    '''
    Parameter
    ----------
    video :
    n_min_views : int
        Minimum number of views per video for the videos to be considered as relevant

    Returns
    ----------
    Boolean which is True if the video has more than 10'000 views
    '''
    try:
        view_counts = video['view_count']
        if view_counts != None:
            return view_counts >= n_min_views
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


def get_freq_tokens_per_video(video, use_bigram, stop_words, tokenizer, stemmer):
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
    stemmed_tokens_per_video = [stemmer.stem(w) for w in tokens_keep]

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
                i_vid % 1000000, j_token)] = freq_tokens_per_video[key]

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
    """
    Parameters
    ----------
    var : any python variable
        variable to be saved
    filename : string
        Name of the pickle we will create
    path_write_data : string
        Path to the folder where we write and keep intermediate results
    """

    with open(path_write_data + filename + '.pickle', 'wb') as f:
        pickle.dump(var, f)
    f.close()


def load_pickle(path_data):
    """
    Parameters:
    ----------
    path_data : string
        Path to the file to be loaded

    Returns
    ----------
    var :
        The loaded object
    """
    with open(path_data, 'rb') as f:
        var = pickle.load(f)
    f.close()

    return var


def get_dataframe_relevant_vid(path_dataset, set_relevant_channels_classifier, n_min_views):
    '''

    Parameters
    ----------
    path_dataset : string
        Path to the folder that contains all files
    set_relevant_channels_classifier : set
        Set of channel_id, which correspond to all the relevant channel for the classifier
    n_min_views : int
        Minimum number of views per video for the videos to be considered as relevant

    Returns
    ----------
    df : pandas.Dataframe
        Dataframe that contains the `channel_id` `view_counts` `uploaded_year` `category` for each relevant videos
    '''
    array_relevant_features = []

    # Reading the file
    with gzip.open(path_dataset, 'rb') as f:

            for i, line in enumerate(f):

                # line is a byte dict, video is the corresponding dict
                video = json.loads(line)

                if check_channel(video, set_relevant_channels_classifier) and check_views(video, n_min_views):

                    array_vid_relevant_features = [video['channel_id'], video['view_count'], video['upload_date'][:4],
                                                   video['categories']]

                    array_relevant_features.append(array_vid_relevant_features)

    # Create Dataframe
    column_names = ['channel_id', 'view_counts', 'uploaded_year', 'category']
    df = pd.DataFrame(array_relevant_features, columns=column_names)

    return df


def get_vocab(path_dataset, set_relevant_channels, n_min_views, use_bigram, stop_words, tokenizer, stemmer):
    '''
    First pass on the whole dataset in order to construct the vocabulary and to
    keep the indexes of all the relevant videos for topic modelling.

    Parameters
    -----------
    path_dataset : string
        Path to the folder that contains all files
    set_relevant_channels : set
        Set of channel_id, which correspond to all the relevant channel
    n_min_views : int
        Minimum number of views per video for the videos to be considered as relevant
    use_bigram : boolean
        if True, use bi_grams in the vocabulary
    stop_words :
    tokenizer :
    stemmer :

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

            if check_channel(video, set_relevant_channels) and check_views(video, n_min_views):
                tokens_per_video = get_freq_tokens_per_video(
                    video, use_bigram, stop_words, tokenizer, stemmer)

                set_relevant_vid.add(i)
                set_stemmed_tokens.update(tokens_per_video)

    return set_stemmed_tokens, set_relevant_vid


def get_document_term_matrix(path_dataset, path_write_data, set_relevant_vid, word2id, use_bigram, stop_words,
                             tokenizer, stemmer):
    """
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
    use_bigram : boolean
        if True, use bi_grams in the vocabulary
    stop_words :
    tokenizer :
    stemmer :

    Returns
    ----------
    S : sparse matrix
        The document-term matrix from the data
    """

    i_vid = 0

    size_of_tokens_dict = len(word2id)
    S = dok_matrix((1000000, size_of_tokens_dict), dtype=np.uint8)
    n_mat_created = 1

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
                    video, use_bigram, stop_words, tokenizer, stemmer)

                # Fill the underlying dict
                dict_freq_tokens_for_sparse_matrix = fill_underlying_dict(
                    freq_tokens_per_video, word2id, i_vid)

                # Fill data in to sparse matrix
                dict.update(S, dict_freq_tokens_for_sparse_matrix)

                # Increase i_vid
                i_vid += 1

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
    """
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
    """
    # create pandas DataFrame of relevant videos with relevant features
    columns_names = ['channel_id', 'view_counts', 'uploaded_year', 'category']

    #  store relevant features of relevant videos in a list
    list_relevant_data = []

    # Read file
    with gzip.open('/dlabdata1/youtube_large/yt_metadata_en.jsonl.gz', 'rb') as f:

        for i, line in enumerate(f):

            if i in set_relevant_vid:
                # line is a str dict, video is the dict corresponding to the str dict
                video = json.loads(line)

                list_vid_relevant_features = [video['channel_id'], video['view_count'], video['upload_date'][:4],
                                              video['categories']]

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
    """
    Parameters
    ----------
    S : sparse matrix
        The document-term matrix from the data
    set_relevant_vid_final : set
        Set of indexes of videos used for topic modelling
    id2word : dict
        Dictionary with id_token as key and the corresponding token as value
    min_vid_per_token : int
        Threshold for tokens that appear in at least min_vid_per_token videos
        to be considered as relevant tokens.

    Returns
    ----------
    S_final : sparse matrix
        The document-term matrix of the data used for topic modelling
    id2word_final : dict
        Dictionary with id_token as key and the corresponding token as value.
        Keep only the filtered tokens
    """

    S_final = S[list(set_relevant_vid_final)]

    # Convert matrix to csc for efficient computing
    S_final = S_final.tocsc()

    list_relevant_tokens = []

    # Iterate on the columns
    for i in range(S_final.shape[1]):
        # Check column has more than 100 non zero entries
        if S_final[:, i].count_nonzero() >= min_vid_per_token:
            list_relevant_tokens.append(i)

    S_final = S_final[:, list_relevant_tokens].tocsr()
    S_final = remove_zero_rows(S_final)

    #  Construct dict of token_id to tokens
    id2word_final = {}

    for i, id_token in enumerate(list_relevant_tokens):
        id2word_final[i] = id2word[id_token]

    return S_final, id2word_final


def get_dict_for_row(row, S):
    """
    Construct SparseVector bag-of-word for each row (videos)

    Parameters
    ----------
    S : sparse matrix
    row : iterable

    Returns
    ----------
    A SparseVector
    """
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


def get_texts(S, id2word):
    '''
    Generate the texts for gensim.models.coherencemodel.CoherenceModel.
    From the document-term matrix and the dictionnary of id_token to token,
    we construct the whole tokenized text, which is a ist of all the tokens
    for each video, represented as a sublist.

    Parameters
    ----------
    S : sparse matrix
        The document-term matrix from the data
    id2word : dict
        Dictionnary with id_token as key and the corresponding token as value

    Returns
    ----------
    texts : list of list
        A list of all the tokens for each video, represented as a sublist
    '''

    texts = []

    for i in range(S.shape[0]):
        token_indices = list(S.getrow(i).nonzero()[1])
        tokens = []

        for token_indice in token_indices:
            tokens.append(id2word[token_indice])
        texts.append(tokens)

    return texts


def get_corpus(S, n_batches=10):
    '''
    Generate the corpus for gensim.models.coherencemodel.CoherenceModel

    Parameters
    ----------
    S : sparse matrix
        The document-term matrix from the data
    n_batches : int
        An integer that allow to proceed the data in batches, for memory consumption

    Returns
    ----------
    corpus : list of list
        Each element in the corpus correspond to a BoW vector of the data
    '''

    corpus = []
    batch_size = int(S.shape[0] / n_batches)

    for k in range(n_batches - 1):

        for row in S[k * batch_size: (k + 1) * batch_size].toarray():
            bow = []
            idx_nonzero = np.nonzero(row)[0]
            for i in range(len(idx_nonzero)):
                bow.append((idx_nonzero[i], row[idx_nonzero[i]]))
            corpus.append(bow)

    for row in S[(n_batches - 1) * batch_size:].toarray():
        bow = []
        idx_nonzero = np.nonzero(row)[0]
        for i in range(len(idx_nonzero)):
            bow.append((idx_nonzero[i], row[idx_nonzero[i]]))
        corpus.append(bow)

    return corpus


def get_coherence_scores(S, spark, path_data, min_ntopic, max_ntopic, texts, corpus, id2word, n_jobs):
    '''
    Get the cv_coherence and umass_coherence_score for the topic models
    from min_ntopic topics to max_ntopic topics.

    Parameters
    ----------
    S : sparse matrix
        The document-term matrix from the data used for topic modelling
    spark : SparkSession
    path_data : string
        Path to the folder that contains all data for coherence model
    min_ntopic : Int
        The number of topics from which the topic coherences will be computed
    max_ntopic : Int
        The number of topics until which the topic coherences will be computed
    texts : list of list
        A list of all the tokens for each video, represented as a sublist
    corpus : list of list
        Each element in the corpus correspond to a BoW vector of the data
    id2word : dict
        A Dictionnary with id_token as key and the corresponding token as value
    n_jobs : int
        Number of cores for parallelization

    Returns
    ----------
    coherence_scores_cv : list
        List of the c_v coherence score for every models
    coherence_scores_umass : list
        List of the u_mass coherence score for every models
    n_topics_list : list
        List of all the number of topics for computing the coherence score of
        the corresponding model
    '''

    coherence_scores_cv = []
    coherence_scores_umass = []

    n_topics_list = np.arange(min_ntopic, max_ntopic + 1, 5)

    for n_topics in n_topics_list:

        # print('Computing coherence score for model with ' +
        #      str(n_topics) + ' topics...')

        # Get describe_topics dataframe
        filename = 'describe_topics_' + str(n_topics) + '.json'

        path_file = path_data + 'describe_topics/' + filename

        describe_topics = spark.read.json(path_file)

        # Characterize the topics with tokens
        topics = []

        for row in describe_topics.sort('topic').rdd.collect():
            tokenized_topic = []
            for j, token_id in enumerate(row.termIndices):
                tokenized_topic.append(id2word[token_id])
                if j > 10:
                    break
            topics.append(tokenized_topic)

        # Compute c_v coherence score and append to coherence scores
        coherence_model = CoherenceModel(topics=topics,
                                         dictionary=FakedGensimDict(
                                             id2word, S),
                                         texts=texts,
                                         coherence='c_v',
                                         processes=n_jobs)

        # Compute u_mass coherence score and append to coherence scores
        coherence_model_umass = CoherenceModel(topics=topics,
                                               corpus=corpus,
                                               dictionary=FakedGensimDict(
                                                   id2word, S),
                                               coherence='u_mass',
                                               processes=n_jobs)

        coherence_scores_cv.append(coherence_model.get_coherence())
        coherence_scores_umass.append(coherence_model_umass.get_coherence())

    return coherence_scores_cv, coherence_scores_umass, n_topics_list


def coherence_plot(coherence_scores_cv, coherence_score_umass, n_topics_list):
    '''Get the plot for coherence scores'''

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_title('Coherence score for a given number of topics', fontsize=24)
    ax.set_xlabel('Number of Topics', fontsize=16)
    ax.set_ylabel('Coherence Score c_v', fontsize=16)

    ax.grid('on')

    ax.plot(coherence_scores_cv, label='c_v coherence score', linewidth=3)

    ax2 = ax.twinx()
    ax2.set_ylabel('Coherence Score u_mass', fontsize=16)
    ax2.plot(coherence_score_umass, label='u_mass coherence score', linewidth=3, color='orange')

    ax.legend(fontsize=16)
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.93), fontsize=16)

    plt.xticks(np.arange(len(n_topics_list)), n_topics_list)

    fig.savefig('coherence_scores')


def get_coherence_scores_tune(S, spark, path_data, n_topics, alphas, betas, texts, corpus, id2word, n_jobs):
    c_v_scores = {}
    u_mass_scores = {}

    for alpha in alphas:
        for beta in betas:

            filename = 'describe_topics_ntopics' + str(n_topics) + '_alpha' + str(alpha) + '_beta' + str(beta) + '.json'
            path_file = path_data + 'describe_topics/tune/' + filename

            # Get the describe_topics dataframe
            describe_topics = spark.read.json(path_file)

            # Characterize the topics with tokens
            topics = []

            for row in describe_topics.sort('topic').rdd.collect():
                tokenized_topic = []
                for j, token_id in enumerate(row.termIndices):
                    tokenized_topic.append(id2word[token_id])
                    if j > 10:
                        break
                topics.append(tokenized_topic)

            # Compute c_v coherence score and append to coherence scores
            coherence_model = CoherenceModel(topics=topics,
                                             corpus=corpus,
                                             dictionary=FakedGensimDict(
                                                 id2word, S),
                                             texts=texts,
                                             coherence='c_v',
                                             processes=n_jobs)

            # Compute u_mass coherence score and append to coherence scores
            coherence_model_umass = CoherenceModel(topics=topics,
                                                   corpus=corpus,
                                                   dictionary=FakedGensimDict(
                                                       id2word, S),
                                                   coherence='u_mass',
                                                   processes=n_jobs)

            c_v_scores[(alpha, beta)] = coherence_model.get_coherence()

            u_mass_scores[(alpha, beta)] = coherence_model_umass.get_coherence()

    return c_v_scores, u_mass_scores


def coherence_plot_tune(n_topics, c_v_scores, u_mass_scores, alphas, betas):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))

    ax1.set_title('C_V coherence score with ' + str(n_topics) + ' topics', fontsize=20, pad=20)
    ax1.set_xlabel('Document Concentration', fontsize=14)
    ax1.set_ylabel('Topic Concentration', fontsize=14)

    im, cbar = heatmap(dict2array(c_v_scores), alphas, betas, ax=ax1,
                       cmap="YlGn", cbarlabel="Coherence Score")
    texts = annotate_heatmap(im, valfmt="{x:.3f} t")

    ax2.set_title('U_mass coherence score with ' + str(n_topics) + ' topics', fontsize=20, pad=20)
    ax2.set_xlabel('Document Concentration', fontsize=14)
    ax2.set_ylabel('Topic Concentration', fontsize=14)

    im2, cbar2 = heatmap(dict2array(u_mass_scores), alphas, betas, ax=ax2,
                         cmap="YlGn", cbarlabel="Coherence Score")
    texts = annotate_heatmap(im2, valfmt="{x:.3f} t")

    plt.show()
    fig.savefig('coherence_scores_tuned_hyperparams')


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=14)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)


    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.3f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def dict2array(dict_):
    """
    Convert a dictionnary with keys as tuple to a matrix where indices are generated from the key tuples

    Parameters
    ----------
    dict_ : dict
        The dictionary as described above

    Returns
    ----------
    Numpy matrix
    """
    array = []
    key_prev = None
    array_sub = []
    for key, val in dict_.items():
        if key[0] != key_prev:
            key_prev = key[0]
            if array_sub:
                array.append(array_sub)
            array_sub = [val]
        else:
            array_sub.append(val)

    array.append(array_sub)

    return np.array(array)


def get_relevant_vid_classifier(df_relevant_vids, n_top_vid_per_combination):
    """
    For the classifier, we select the documents that are relevant except the ones
    used for training the topic modelling model.

    Parameters
    ----------
    df_relevant_vids : pandas.core.frame.DataFrame
        Dataframe that contains the relevant features of the videos
    n_top_vid_per_combination : int
        Threshold for selecting the number videos with the most views for each combination of
        \'category\', \'uploaded_year\' and \'channel_id\' for topic modelling

    Returns
    ----------
    df_relevant_vid_classifier : pandas.core.frame.DataFrame
        Dataframe that contains the relevant features of the videos selected for the classifier which are the relevant
        videos that were not used for training the topic modelling model.
    """
    df_used_data = df_relevant_vids.sort_values(['view_counts'], ascending=False).groupby(['category', 'uploaded_year', 'channel_id']).head(n_top_vid_per_combination)
    index_to_remove = set(df_used_data.index)

    set_relevant_vid_classifier = set()

    for index in df_relevant_vids.index:
        if index not in index_to_remove:
            set_relevant_vid_classifier.add(index)

    df_relevant_vid_classifier = df_relevant_vids.iloc[list(set_relevant_vid_classifier)]

    return df_relevant_vid_classifier


def get_balanced_data_for_classifier(df_relevant_vids_classifier):
    """
    Parameters
    ----------
    df_relevant_vid_classifier : pandas.core.frame.DataFrame
        Dataframe that contains the relevant features of the videos selected for the classifier which are the relevant
        videos that were not used for training the topic modelling model.

    Returns
    ----------
    index_balanced_videos_classifier : set
        Set of the indices of videos for the classifier. The videos from the indices are categories-balanced in order
        to avoid all bias while measuring the model with the classifier.
    """

    category2idx = {}

    for index, elem in zip(df_relevant_vids_classifier.index, df_relevant_vids_classifier['category']):
        if elem in category2idx.keys():
            category2idx[elem].append(index)
        else:
            category2idx[elem] = [index]

    # Find the category with the lowest number of videos and keep this number
    min_n_videos = math.inf

    for key, value in category2idx.items():
        if key != '' and key != 'Shows':
            min_n_videos = min(min_n_videos, len(value))

    # Randomly select min_n_videos indexes of each video's category
    category2idx_rand_undersampl = {}

    for key, value in category2idx.items():
        category2idx_rand_undersampl[key] = random.choices(value, k=int(min_n_videos / 10))

    # Create our final array of index
    index_balanced_videos_classifier = []
    for key, value in category2idx_rand_undersampl.items():
        if key != '' and key != 'Shows':
            index_balanced_videos_classifier.extend(value)

    return set(index_balanced_videos_classifier)


def get_data_for_classifier(id2word_final, set_relevant_vids_classifier, stop_words, tokenizer, stemmer):
    """
    Parameters
    ----------
    id2word_final : dict
    set_relevant_vids_classifier : set
    stop_words :
    tokenizer :
    stemmer :


    Returns
    ----------
    S : sparse matrix
    groundtruth : list

    """

    # Get the vocabulary of the TM model
    vocab = list(id2word_final.values())

    # we need word2id in order to be able to match the token to its index in the vocabulary
    word2id = {v: k for k, v in id2word_final.items()}

    S = dok_matrix((len(set_relevant_vids_classifier), len(vocab)), dtype=np.uint8)
    groundtruth = []
    i_vid = 0

    with gzip.open('/dlabdata1/youtube_large/yt_metadata_en.jsonl.gz', 'rb') as f:

        for i, line in enumerate(f):

            if i in set_relevant_vids_classifier:

                # line is a str dict, video is the dict corresponding to the str dict
                video = json.loads(line)

                # Get the tokens for each video and theirs number of occurrences
                freq_tokens_per_video = get_freq_tokens_per_video(video, False, stop_words, tokenizer, stemmer)

                # For each video, create a underlying dictionary for filling the sparse matrix efficiently
                dict_freq_tokens_for_sparse_matrix = fill_underlying_dict(freq_tokens_per_video, word2id, i_vid)

                # Need to check that the video contains token from the reduced vocabulary
                if dict_freq_tokens_for_sparse_matrix != {}:
                    # Update the Sparse Matrix
                    dict.update(S, dict_freq_tokens_for_sparse_matrix)
                    i_vid += 1

                    # Get groundtruth values
                    groundtruth.append(video['categories'])

    # Save last sparse matrix
    S = S.tocsr()
    S = remove_zero_rows(S)

    return S, groundtruth


def get_doc_topic_matrices(path_model, df_data, n_docs, vocabSize):
    """
    Parameters
    ----------
    path_model : string
    df_data : pyspark.sql.dataframe.DataFrame
    n_docs : int
    vocabSize : int

    Returns
    ----------
    doc_topic_matrix : dense matrix in csr_matrix format
        Matrix of the distribution over the topics for each document
    """
    model = LocalLDAModel.load(path_model)
    transformed_data = model.transform(df_data)

    doc_topic_matrix = dok_matrix((n_docs, vocabSize))

    for i, topic_dist_one_vid in enumerate(transformed_data.select('topicDistribution').collect()):

        dict_topic_dist_one_vid = {}

        for j, prob in enumerate(topic_dist_one_vid['topicDistribution']):
            dict_topic_dist_one_vid[(i, j)] = prob

        # Fill data in to sparse matrix
        dict.update(doc_topic_matrix, dict_topic_dist_one_vid)

    return doc_topic_matrix.tocsr()


def split_train_test_val(path_dataset, train_size, n_docs):
    """

    Parameters
    ----------
    path_dataset : string
        Path to the folder that contains all files
    train_size : float
    n_docs : int
        The total number of documents


    Returns
    ----------
    list_train_idx : list
    list_test_idx : list
    list_val_idx : list
    """

    path_index = path_dataset + 'index_train_test_val_clasifier.pickle'
    if os.path.isfile(path_index):
        index = load_pickle(path_index)
    else:
        index = np.arange(n_docs)
        np.random.shuffle(index)
        save_to_pickle(index, 'index_train_test_val_clasifier', path_dataset)

    training_val_model_threshold = int(train_size * n_docs)
    training_model_threshold = int(0.9 * train_size * n_docs)

    list_train_idx = np.sort(index[:training_model_threshold])
    list_val_idx = np.sort(index[training_model_threshold:training_val_model_threshold])
    list_test_idx = np.sort(index[training_val_model_threshold:])

    return list_train_idx, list_test_idx, list_val_idx


def classifier(doc_topic_matrix, groundtruth, reg_coefs, list_train_idx, list_test_idx, list_val_idx):
    """

    :param doc_topic_matrix:
    :param groundtruth:
    :param reg_coefs:
    :param list_train_idx:
    :param list_test_idx:
    :param list_val_idx:
    :return:
    """

    # Train a classifier by tuning the hyperparameter for the regularization term
    accuracies_tmp = []

    for reg_coef in reg_coefs:

        clf = SGDClassifier(loss='hinge', alpha=reg_coef, max_iter=100, shuffle=True, n_jobs=10, random_state=1)
        clf.fit(doc_topic_matrix[list_train_idx], y=np.array(groundtruth)[list_train_idx])

        y_pred = clf.predict(doc_topic_matrix[list_val_idx])
        y_gt = np.array(groundtruth)[list_val_idx]

        score = accuracy_score(y_gt, y_pred)
        accuracies_tmp.append(score)

    best_reg_coef = reg_coefs[np.argmax(accuracies_tmp)]

    # apply the best classifier to test data
    clf = SGDClassifier(loss='hinge', alpha=best_reg_coef, max_iter=100, shuffle=True, n_jobs=10, random_state=1)
    clf.fit(doc_topic_matrix[np.concatenate([list_train_idx, list_val_idx])],
            y=np.array(groundtruth)[np.concatenate([list_train_idx, list_val_idx])])

    y_pred = clf.predict(doc_topic_matrix[list_test_idx])
    y_gt = np.array(groundtruth)[list_test_idx]

    accuracy = accuracy_score(y_gt, y_pred)

    return accuracy


def plot_accuracy_classifier(accuracies, n_topics):
    """

    :param accuracies:
    :param n_topics:
    :return:
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.set_title('Classifier accuracy for a clustering model with k number of topics', fontsize=24, pad=10)
    ax.set_xlabel('Number of Topics', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)

    ax.grid('on')

    ax.plot(accuracies, label='accuracy score', linewidth=3)

    ax.legend(fontsize=16)

    plt.xticks(np.arange(len(n_topics)), n_topics)

    fig.savefig('accuracies_for_n_topics')


def plot_accuracy_classifier_tune(accuracies, alphas, betas):
    """

    :param accuracies:
    :param alphas:
    :param betas:
    :return:
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title('Accuracy score with 55 topics', fontsize=20, pad=20)
    ax.set_xlabel('Document Concentration', fontsize=14)
    ax.set_ylabel('Topic Concentration', fontsize=14)

    im, cbar = heatmap(np.array(accuracies), alphas, betas, ax=ax,
                       cmap="YlGn", cbarlabel="Accuracy Score")
    texts = annotate_heatmap(im, valfmt="{x:.3f} t")

    # fig.tight_layout()
    plt.show()
    fig.savefig('accuracies_for_alphas_betas')


def get_data_for_LDA(path_data, spark):
    """
    Parameters
    ----------
    path_data : string
        Path to the data generated by data_processing.py
    spark :

    Returns
    ----------
    df_data : pyspark.sql.dataframe.DataFrame
        DataFrame for input of the LDA model
    vocabSize : int
        Size of the vocabulary from the documents
    """

    df_load = spark.read.json(path_data)
    data = []
    vocabSize = 0

    for i, row in enumerate(df_load.collect()):

        features = row['features']
        data.append([row['id'], SparseVector(
            features['size'], features['indices'], features['values'])])

        if i == 0:
            vocabSize = features['size']

    df_data = spark.createDataFrame(data, ['id', 'features'])

    return df_data, vocabSize