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

    all_data = []

    print('Process video for topic modelling...')
    for i in range(S.shape[0]):

        if i % 1000000 == 0:
            print(str(i) + ' videos processed...')

        all_data.append([i, get_dict_for_row(S.getrow(i).todok().items(), S)])

    # Tuning LDAModel
    n_topics = 100

    # Construct dataframe for LDA
    all_df = spark.createDataFrame(all_data, ["id", "features"])

    # Build LDA model
    print('Computing with ' + str(n_topics) + ' topics...')
    lda = LDA(k=n_topics, seed=1)
    model = lda.fit(all_df)

    # save the model attributes
    print('Save the model attributes...')

    model.describeTopics(maxTermsPerTopic=10).write\
                    .option('compression', 'gzip')\
                    .json('/dlabdata1/youtube_large/olam/LDA_Model/testModel/describe_topics.json')

    topic_columns = []
    for i in range(n_topics):
        topic_columns.append('Topic' + str(i))

    spark.createDataFrame(model.topicsMatrix().toArray().tolist(), topic_columns)\
                    .write\
                    .option('compression', 'gzip')\
                    .json('/dlabdata1/youtube_large/olam/LDA_Model/testModel/topics_term_matrix.json')

    model.transform(all_df).write\
                    .option('compression', 'gzip')\
                    .json('/dlabdata1/youtube_large/olam/LDA_Model/testModel/topics_doc_matrix.json')



    print('Task done!')

if __name__ == "__main__":
    main()
