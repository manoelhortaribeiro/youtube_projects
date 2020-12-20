import argparse
import numpy as np
import pickle
import random
import sys
import time

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import LDA, LDAModel, LocalLDAModel
from pyspark.ml.linalg import Vectors, SparseVector


def main():
    conf = SparkConf()

    print('Using bigrams: ' + str(use_bigram))

    if use_bigram:
        path_data = '/user/olam/with_bigram/'
        vocabSize = 97888
    else:
        path_data = '/user/olam/final_res/'
        vocabSize = 53255

    # create the session
    spark = SparkSession.builder.appName(
        "LDA_topicmodelling").config(conf=conf).getOrCreate()

    # Load data
    n_topics_list = [50, 55, 60, 65, 70]

    max_iter = 1000

    print('Initializing... Tuning the number of topics with ' +
          str(max_iter) + ' iters...')

    print('Loading data...')
    df_load_view10000_sub100000 = spark.read.json(path_data + 'sparkdf.json')

    # Process data
    data_view10000_sub100000 = []

    print('Processing data...')
    for row in df_load_view10000_sub100000.collect():
        features = row['features']
        data_view10000_sub100000.append([row['id'], SparseVector(
            features['size'], features['indices'], features['values'])])

    df_view10000_sub100000 = spark.createDataFrame(
        data_view10000_sub100000, ['id', 'features'])

    for n_topics in n_topics_list:

        print('Training with ' + str(n_topics) + ' topics...')
        lda_view10000_sub100000 = LDA(k=n_topics, maxIter=max_iter, seed=1)
        model_view10000_sub100000 = lda_view10000_sub100000.fit(
            df_view10000_sub100000)

        # save the model attributes
        print('Save the model attributes...')

        model_view10000_sub100000.describeTopics(maxTermsPerTopic=vocabSize).write\
            .option('compression', 'gzip')\
            .json(path_data + 'describe_topics_' + str(n_topics) + '.json')

        model_view10000_sub100000.transform(df_view10000_sub100000).write\
            .option('compression', 'gzip')\
            .json(path_data + 'topics_doc_matrix_' + str(n_topics) + 'json')

        print('Save the model...')

        model_view10000_sub100000.save(path_data + 'model_' + str(n_topics))

    print('Task done!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LDA Tune')
    parser.add_argument('--use_bigram', dest='use_bigram',
                        default=False, action='store_true')

    args = parser.parse_args()

    use_bigram = args.use_bigram

    main()
