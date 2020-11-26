import argparse
import numpy as np
import os
import pickle
import random
import scipy.sparse
import sys
import time

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.clustering import LDA, LDAModel, LocalLDAModel
from pyspark.ml.linalg import Vectors, SparseVector
from scipy.sparse import dok_matrix


def main():

    print('Get the data...')
    with open('/dlabdata1/youtube_large/olam/data/final_res/classifier/data_spark0.pickle', 'rb') as f:
        data = pickle.load(f)
    f.close()

    for i in range(1, 9):
        print('Processing file ' + str(i))
        filename = 'data_spark' + str(i) + '.pickle'
        with open('/dlabdata1/youtube_large/olam/data/final_res/classifier/' + filename, 'rb') as f:
            data.extend(pickle.load(f))
        f.close()

    with open('/dlabdata1/youtube_large/olam/data/final_res/classifier/data_spark_last.pickle', 'rb') as f:
        data.extend(pickle.load(f))
    f.close()

    print('Initialize spark session...')
    conf = SparkConf().setMaster("local[4]").setAll(
        [('spark.executor.memory', '8g'), ('spark.driver.memory', '64g'), ('spark.driver.maxResultSize', '0')])

    # create the session
    spark = SparkSession.builder.appName(
        "LDA_topicmodelling").config(conf=conf).getOrCreate()

    print('Create spark dataframe...')
    df = spark.createDataFrame(data, ["id", "features"])

    # Get model
    print('Load the model...')
    model_name = 'model_' + n_topic
    model = LocalLDAModel.load(
        '/dlabdata1/youtube_large/olam/data/final_res/models/' + model_name)

    # Get the transformed data
    print('Get the transformed data')
    transformed_data = model.transform(df)

    number_videos_in_dataset = len(data)
    batch_size = 500000
    n_iter = int(number_videos_in_dataset / batch_size)

    if not os.path.exists('/dlabdata1/youtube_large/olam/data/final_res/matrices/transformed_data' + n_topic):
        os.mkdir(
            '/dlabdata1/youtube_large/olam/data/final_res/matrices/transformed_data' + n_topic)

    print('Get the transformed data to small scipy sparse matrices...')
    for k in range(n_iter):

        print('Iteration ' + str(k + 1) + '/' + str(n_iter))

        transformed_data_sub = transformed_data.where(
            col("id").between(0 + k * 500000, (k + 1) * 500000 - 1))

        # Create sparse matrix
        S = dok_matrix((batch_size, int(n_topic)))

        for i, topic_dist_one_vid in enumerate(transformed_data_sub.select('topicDistribution').collect()):

            dict_topic_dist_one_vid = {}

            for j, prob in enumerate(topic_dist_one_vid['topicDistribution']):

                dict_topic_dist_one_vid[(i, j)] = prob

            # Fill data in to sparse matrix
            dict.update(S, dict_topic_dist_one_vid)

        filename = 'transformed_data' + str(k) + '.npz'
        scipy.sparse.save_npz(
            '/dlabdata1/youtube_large/olam/data/final_res/matrices/transformed_data' + n_topic + '/' + filename, S.tocsr())

    # last iteration
    transformed_data_sub = transformed_data.where(
        col("id").between(0 + n_iter * 500000, number_videos_in_dataset + 1))

    S = dok_matrix((number_videos_in_dataset -
                    batch_size * n_iter, int(n_topic)))

    for i, topic_dist_one_vid in enumerate(transformed_data_sub.select('topicDistribution').collect()):

        dict_topic_dist_one_vid = {}

        for j, prob in enumerate(topic_dist_one_vid['topicDistribution']):

            dict_topic_dist_one_vid[(i, j)] = prob

        # Fill data in to sparse matrix
        dict.update(S, dict_topic_dist_one_vid)

    filename = 'transformed_data_last.npz'
    scipy.sparse.save_npz(
        '/dlabdata1/youtube_large/olam/data/final_res/matrices/transformed_data' + n_topic + '/' + filename, S.tocsr())

    # Get the full matrix
    print('Load all the matrices to get one big final matrix...')
    data = scipy.sparse.load_npz(
        '/dlabdata1/youtube_large/olam/data/final_res/matrices/transformed_data' + n_topic + '/transformed_data0.npz')

    for i in range(1, 38):
        data_next = scipy.sparse.load_npz(
            '/dlabdata1/youtube_large/olam/data/final_res/matrices/transformed_data' + n_topic + '/transformed_data' + str(i) + '.npz')
        data = scipy.sparse.vstack([data, data_next])

    # Add last matrix
    data_next = scipy.sparse.load_npz(
        '/dlabdata1/youtube_large/olam/data/final_res/matrices/transformed_data' + n_topic + '/transformed_data_last.npz')
    data = scipy.sparse.vstack([data, data_next])

    # Save the full matrix
    scipy.sparse.save_npz(
        '/dlabdata1/youtube_large/olam/data/final_res/matrices/transformed_data' + n_topic + '/transformed_data_final.npz', data)

    print('Task done!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get_transformed_data')
    parser.add_argument('--n_topic', dest='n_topic', type=int)

    args = parser.parse_args()

    n_topic = str(args.n_topic)

    main()
