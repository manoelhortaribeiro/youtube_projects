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

    print('Tuning the alphas for the given beta = ' + str(beta) + '...')

    conf = SparkConf()

    # create the session
    spark = SparkSession.builder.appName(
        "LDA Hyperparameter Tuning").config(conf=conf).getOrCreate()

    # Load data
    alphas = [0.1, 0.5, 0.9, 0.95, 1]
    max_iter = 500
    vocabSize = 53255

    print('Loading data...')
    df_load = spark.read.json(
        '/user/olam/final_res/sparkdf.json')

    # Process data
    data = []

    print('Processing data...')
    for row in df_load.collect():
        features = row['features']
        data.append([row['id'], SparseVector(
            features['size'], features['indices'], features['values'])])

    df = spark.createDataFrame(
        data, ['id', 'features'])

    for alpha in alphas:

        print('Training with alpha = ' + str(alpha) + '...')
        lda = LDA(
            k=n_topics, maxIter=max_iter, seed=1, docConcentration=[alpha], topicConcentration=beta)
        model = lda.fit(df)

        # save the model attributes
        print('Save the model attributes...')

        name_suffixes = 'ntopics' + \
            str(n_topics) + '_alpha' + str(alpha) + '_beta' + str(beta)

        model.describeTopics(maxTermsPerTopic=vocabSize).write\
            .option('compression', 'gzip')\
            .json('/user/olam/final_res/tune/describe_topics_' + name_suffixes + '.json')

        model.transform(df).write\
            .option('compression', 'gzip')\
            .json('/user/olam/final_res/tune/topics_doc_matrix_' + name_suffixes + 'json')

        print('Save the model...')

        model.save(
            '/user/olam/final_res/tune/model_' + name_suffixes)

    print('Task done!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Topic Concentration')
    parser.add_argument('--ntopic', dest='ntopic', type=int)
    parser.add_argument('--beta', dest='beta', type=float)

    args = parser.parse_args()

    n_topics = args.ntopic
    beta = args.beta

    main()
