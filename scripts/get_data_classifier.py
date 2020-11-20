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
    spark = SparkSession.builder.appName(
        "LDA_topicmodelling").config(conf=conf).getOrCreate()

    print('Get the data...')
    with open('/user/olam/final_res/classifier/data_spark0.pickle', 'rb') as f:
        data = pickle.load(f)
    f.close()

    for i in range(1, 9):
        filename = 'data_spark' + str(i) + '.pickle'
        with open('/user/olam/final_res/classifier/' + filename, 'rb') as f:
            data.extend(pickle.load(f))
        f.close()

    with open('/user/olam/final_res/classifier/data_spark_last.pickle', 'rb') as f:
        data.extend(pickle.load(f))
    f.close()

    df = spark.createDataFrame(data_spark, ["id", "features"])

    print('Load the already computed model...')
    # Load the model
    model = LocalLDAModel.load('/user/olam/final_res/model_110')

    print('Get the distribution over the topics...')
    # Get the transformed data
    transformed_data = model.transform(df)

    print('Save the transformed data...')
    transformed_data.write.option('compression', 'gzip').json(
        '/user/olam/final_res/classifier/transformed_data110.json')

    print('Task done!')


if __name__ == "__main__":
    main()
