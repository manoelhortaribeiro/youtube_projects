import argparse
import os

import numpy as np

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import LDA
from pyspark.ml.linalg import SparseVector


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


def main():

    # Some checks with the given arguments
    if args.tune and args.n_topic_tune is None:
        raise ValueError('The number of topics is required when tuning the docConcentration and topicConcentration.')

    if args.tune is False and (args.min_n_topic is None or args.max_n_topic is None):
        raise ValueError('When not tuning the docConcentration and topicConcentration, min_n_topic and max_n_topic are required')

    # Create a Spark Session for reading the data
    conf = SparkConf().set("spark.driver.maxResultSize", "0")
    spark = SparkSession.builder.appName(
        "LDA_topicmodelling").config(conf=conf).getOrCreate()

    # Get the right data for LDA
    print('Generating data for LDA...')
    df, vocabSize = get_data_for_LDA(args.path_data + 'sparkdf.json', spark)

    if args.tune:

        # Create the folders for storing results if needed
        path_describe_topics_tune = args.path_data + 'describe_topics_tune'
        path_topics_doc_matrix_tune = args.path_data + 'topics_doc_matrix_tune'
        path_models_tune = args.path_data + 'models_tune'

        # Initialize range of hyperparameters
        alphas = [0.1, 0.5, 0.9, 0.95, 1]
        betas = [0.01, 0.05, 0.1, 0.5]

        for alpha in alphas:
            for beta in betas:

                # Compute the model
                print('Computing the model with ' + str(args.n_topic_tune) + ' topics with docConcentration = ' + str(alpha) + ' and topicConcentration = ' + str(beta))
                lda = LDA(k=args.n_topic_tune, maxIter=args.n_iter, seed=1, docConcentration=[alpha], topicConcentration=beta)
                model = lda.fit(df)

                # Save the model
                print('Saving the model and its data')
                path_describeTopics = path_describe_topics_tune + '/describe_topics_' + str(args.n_topic_tune) + '_alpha' + str(alpha) + '_beta' + str(beta)
                model.describeTopics(maxTermsPerTopic=vocabSize).write.option('compression', 'gzip').json(
                    path_describeTopics)

                path_transformed_data = path_topics_doc_matrix_tune + '/topics_doc_matrix_' + str(args.n_topic_tune) + '_alpha' + str(alpha) + '_beta' + str(beta)
                model.transform(df).write.option('compression', 'gzip').json(path_transformed_data)

                path_model = path_models_tune + '/model_' + str(args.n_topic_tune) + '_alpha' + str(alpha) + '_beta' + str(beta)
                model.save(path_model)

    else:

        # Create the folders for storing results if needed
        path_describe_topics = args.path_data + 'describe_topics'
        path_topics_doc_matrix = args.path_data + 'topics_doc_matrix'
        path_models = args.path_data + 'models'

        # Create array of all topics for LDA
        n_topics_array = np.arange(args.min_n_topic, args.max_n_topic + 1, 5)

        for n_topics in n_topics_array:

            # Compute the model
            print('Computing the model with ' + str(n_topics) + ' topics...')
            lda = LDA(k=n_topics, maxIter=args.n_iter, seed=1)
            model = lda.fit(df)

            # Save the model
            print('Saving the model and its data')
            path_describeTopics = path_describe_topics + '/describe_topics_' + str(n_topics)
            model.describeTopics(maxTermsPerTopic=vocabSize).write.option('compression', 'gzip').json(path_describeTopics)

            path_transformed_data = path_topics_doc_matrix + '/topics_doc_matrix_' + str(n_topics)
            model.transform(df).write.option('compression', 'gzip').json(path_transformed_data)

            path_model = path_models + '/model_' + str(n_topics)
            model.save(path_model)

    print('Task Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Arguments for tuning the number k of topics
    parser.add_argument('--min_n_topic', dest='min_n_topic', type=int, default=None,
                        help='The minimum number of topics for the LDA model. Multiples models will be computer from the min_n_topic to max_n_topic, with an increase of 5 by 5')
    parser.add_argument('--max_n_topic', dest='max_n_topic', type=int, default=None,
                        help='The maximum number of topics for the LDA model. Multiples models will be computer from the min_n_topic to max_n_topic, with an increase of 5 by 5')

    # Arguments for tuning hyperparameters of LDA given number of topic
    parser.add_argument('--tune', dest='tune', default=False, action='store_true',
                        help='Boolean that indicate that tuning the docConcentration and topicConcentration will be performed. Required n_topic_tune.')
    parser.add_argument('--n_topic_tune', dest='n_topic_tune', type=int, default=None,
                        help='The number of topics on which docConcentration and topicConcentration will be tuned.')

    # General arguments
    parser.add_argument('--path_data', dest='path_data', default='/user/olam/test/')
    parser.add_argument('--n_iter', dest='n_iter', type=int, default=500,
                        help='The number of iterations for the LDA algorithm')

    args = parser.parse_args()

    main()
