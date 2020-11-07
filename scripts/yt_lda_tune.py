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

    # create the session
    spark = SparkSession.builder.appName("LDA_topicmodelling").config(conf=conf).getOrCreate()

    # Load data
    n_topics_list = [155, 160, 165, 170, 175]
    max_iter = 1000
    vocabSize = 42757

    print('Initializing... Tuning the number of topics with ' + str(max_iter)  + ' iters')

    print('Loading data...')
    df_load_view10000_sub100000 = spark.read.json('/user/olam/view10000_sub100000/sparkdf.json')

    # Process data
    data_view10000_sub100000 = []

    print('Processing data...')
    for row in df_load_view10000_sub100000.collect():
        features = row['features']
        data_view10000_sub100000.append([row['id'], SparseVector(features['size'], features['indices'], features['values'])])

    df_view10000_sub100000 = spark.createDataFrame(data_view10000_sub100000, ['id', 'features'])

    for n_topics in n_topics_list:

        print('Training with ' + str(n_topics) + ' topics...')
        lda_view10000_sub100000 = LDA(k=n_topics, maxIter=max_iter ,seed=1)
        model_view10000_sub100000 = lda_view10000_sub100000.fit(df_view10000_sub100000)

        # save the model attributes
        print('Save the model attributes...')

        model_view10000_sub100000.describeTopics(maxTermsPerTopic=vocabSize).write\
                        .option('compression', 'gzip')\
                        .json('/user/olam/view10000_sub100000/tune/describe_topics_' + str(n_topics) + '_iter1000_tok100vid.json')

        model_view10000_sub100000.transform(df_view10000_sub100000).write\
                        .option('compression', 'gzip')\
                        .json('/user/olam/view10000_sub100000/tune/topics_doc_matrix_' + str(n_topics) + '_iter1000_tok100vid.json')


    print('Task done!')

if __name__ == "__main__":
    main()
