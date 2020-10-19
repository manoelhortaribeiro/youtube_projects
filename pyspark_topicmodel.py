import numpy as np
import pickle
import random
import scipy.sparse
import sys
import time

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import LDA, LDAModel, LocalLDAModel
from pyspark.ml.linalg import Vectors, SparseVector



def get_dict_for_row(row, S):
    '''Construct SparseVector bag-of-word for each row (videos)'''
    tmp_dict = {}
    for key, value in row:
        tmp_dict[key[1]] = value

    return SparseVector(S.shape[1], tmp_dict)


def remove_col(S):
    '''Remove all the unseen tokens from the data given S'''

    # First remove unseen tokens
    S = S.tocsc()

    S_keep_idx = []

    for i in range(S.shape[1]):
        if S[:,i].count_nonzero()!= 0:
            S_keep_idx.append(i)

    return S[:,S_keep_idx].tocsr(), S_keep_idx


def main():
    conf = SparkConf().setMaster("local[16]").setAll([
     ('spark.executor.memory', '8g'),  # find
     ('spark.driver.memory','64g'), # your
     ('spark.driver.maxResultSize', '0') # setup
    ])

    # create the session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # create the context
    sc = spark.sparkContext

    # Load data
    print('Loading data...')
    S = scipy.sparse.load_npz('/dlabdata1/youtube_large/olam/matrices/S_final3.npz')

    # We want to tune the optimal number of topics => find it on random subset of videos
    print('Getting subsample of videos...')
    id_vid2train = random.sample(range(0,S.shape[0]), 5000000)
    S_sub = S[id_vid2train,:]
    S_sub, S_sub_tokens_id = remove_col(S_sub)

    # Split subset of videos to training and testing dataset
    train_data_idx = set(random.sample(range(0,S_sub.shape[0]), int(0.8*S_sub.shape[0])))

    all_data = []
    train_data = []
    test_data = []

    train_data_idx_sorted = 0
    test_data_idx_sorted = 0

    print('Process video for topic modelling...')
    for i in range(S_sub.shape[0]):

        if i % 1000000 == 0:
            print(str(i) + ' videos processed...')

        all_data.append([i, get_dict_for_row(S_sub.getrow(i).todok().items(), S_sub)])

        # Data is a list of list of the following elems : index of doc and a bag-of-word sparse Vector
        if i in train_data_idx:
            train_data.append([train_data_idx_sorted, get_dict_for_row(S_sub.getrow(i).todok().items(), S_sub)])
            train_data_idx_sorted += 1
        else:
            test_data.append([test_data_idx_sorted, get_dict_for_row(S_sub.getrow(i).todok().items(), S_sub)])
            test_data_idx_sorted += 1


    # Tuning LDAModel
    numbers_topics = [25]
    perplex_scores = []
    models = []

    # Construct dataframe for LDA
    all_df = spark.createDataFrame(all_data, ["id", "features"])
    train_df = spark.createDataFrame(train_data, ["id", "features"])
    test_df = spark.createDataFrame(test_data, ["id", "features"])

    # Tuning by minimizing the perplexity score
    for n_topic in numbers_topics:
        print('Computing with ' + str(n_topic) + ' topics...')
        lda = LDA(k=n_topic, seed=1)
        model = lda.fit(train_df)
        print('Computing perplexity for model with ' + str(n_topic) + ' topics...')
        #logperplexity = model.logPerplexity(test_df)

        models.append(model)
        #perplex_scores.append(logperplexity)


    n_topics_opt = 25 #numbers_topics[np.argmin(perplex_scores)]
    print('The optimal choice for the number of topics : ' + str(n_topics_opt))

    # Save the perplexity scores and the model
    with open('/dlabdata1/youtube_large/olam/LDA_Model/testModel/perplex_scores_n_topic_optimal.pickle', 'wb') as f:
        pickle.dump({'perplex_scores':perplex_scores, 'n_topics_opt':n_topics_opt}, f)
    f.close()


    # Fitting the model with n_topics_opt in order to visualise the results
    print('Fit best model with all data for inspection...')
    lda = LDA(k=n_topics_opt, seed=1)
    model = lda.fit(all_df)

    # save the model attributes
    print('Save the model attributes...')

    model.describeTopics(maxTermsPerTopic=10).write\
                    .option('compression', 'gzip')\
                    .json('/dlabdata1/youtube_large/olam/LDA_Model/testModel/describe_topics.json')

    topic_columns = []
    for i in range(n_topics_opt):
        topic_columns.append('Topic' + str(i))

    spark.createDataFrame(model.topicsMatrix().toArray().tolist(), topic_columns)\
                    .write\
                    .option('compression', 'gzip')\
                    .json('/dlabdata1/youtube_large/olam/LDA_Model/testModel/topics_term_matrix.json')

    model.transform(all_df).write\
                    .option('compression', 'gzip')\
                    .json('/dlabdata1/youtube_large/olam/LDA_Model/testModel/topics_doc_matrix.json')


    # save the data that has been used for tuning
    scipy.sparse.save_npz('/dlabdata1/youtube_large/olam/LDA_Model/testModel/S_sub.npz', S_sub)

    with open('/dlabdata1/youtube_large/olam/LDA_Model/testModel/S_sub_tokens_id.pickle', 'wb') as f:
        pickle.dump(S_sub_tokens_id, f)
    f.close()


    print('Task done!')

if __name__ == "__main__":
    main()
