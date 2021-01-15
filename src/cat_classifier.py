import argparse
import os
import scipy

import numpy as np

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from utils import *


def main():
    # Some checks with the given arguments
    if args.tune and args.n_topic_tune is None:
        raise ValueError('The number of topics is required when building classifiers for LDA models with specific values of docConcentration and topicConcentration.')

    if args.tune is False and (args.min_n_topic is None or args.max_n_topic is None):
        raise ValueError(
            'min_n_topic and max_n_topic are required when building classifiers for LDA models with different number of topics')

    # Setting NLP pre-processing features
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer(language='english')

    # Make directory for storing intermediate results for classifier
    path_doc_topic_mat = args.path_write_data + 'matrices/doc_topic_mat'
    if not os.path.isdir(path_doc_topic_mat):
        os.mkdir(path_doc_topic_mat)
    del path_doc_topic_mat

    # Load the dictionary of indices to tokens from the processing dataset to match the new videos to that vocabulary
    id2word_final = load_pickle(args.path_write_data + 'id2word_final.pickle')

    if os.path.isdir(args.path_write_data + 'models/df_balanced_data.json'):

        # Create SparkSession
        print('Loading data...')
        spark, conf = create_spark_session(args.n_jobs, args.executor_mem, args.driver_mem)
        df_balanced_data, _ = get_data_for_LDA(args.path_write_data + 'models/df_balanced_data.json', spark)
        S = scipy.sparse.load_npz(args.path_write_data + 'matrices/S_balanced_for_classifier.npz')
        groundtruth = load_pickle(args.path_write_data + 'groundtruth.pickle')

    else:
        print('Generating the data for classifier...')
        if args.n_min_sub_classifier is None:
            set_relevant_channels = load_pickle(args.path_write_data + 'set_relevant_channels.pickle')
        else:
            set_relevant_channels = get_relevant_channels(args.n_min_sub_classifier)

        # Get the relevant vids
        df_relevant_vids = get_dataframe_relevant_vid(args.path_dataset, set_relevant_channels, args.n_min_views_classifier)

        # Remove the videos that have been used for training the LDA model
        df_relevant_vids_classifier = get_relevant_vid_classifier(df_relevant_vids, args.n_top_vid_per_combination)

        # Get the indexes of the videos for the classifier.
        # We need to ensure that the categories of these videos are well-balanced
        print('Get balanced data...')
        index_balanced_videos_classifier = get_balanced_data_for_classifier(df_relevant_vids_classifier)

        # Get the data for the videos with the corresponding groundtruth, which is the category of the video
        S, groundtruth = get_data_for_classifier(id2word_final, index_balanced_videos_classifier, stop_words, tokenizer, stemmer)

        # Create SparkSession
        spark, conf = create_spark_session(args.n_jobs, args.executor_mem, args.driver_mem)

        # Get the data ready for transforming it with the LDA models
        df_balanced_data = get_data_for_tm(S, spark)

        # Save intermediate results
        print('Saving intermediate results...')
        df_balanced_data.write.option('compression', 'gzip').json(args.path_write_data + 'models/df_balanced_data.json')
        scipy.sparse.save_npz(args.path_write_data + 'matrices/S_balanced_for_classifier.npz', S)
        save_to_pickle(groundtruth, 'groundtruth', args.path_write_data)


    # get index for training, testing and validation dataset for the classifier
    list_train_idx, list_test_idx, list_val_idx = split_train_test_val(args.path_write_data, train_size=0.8, n_docs=len(groundtruth))
    reg_coefs = np.logspace(0, 1, num=20, base=100) / 100

    if args.tune:

        print('Train classifiers on models with defined hyperparameters...')

        # Initialize range of hyperparameters
        alphas = [0.1, 0.5, 0.9, 0.95, 1]
        betas = [0.01, 0.05, 0.1, 0.5]

        accuracies = []

        for alpha in alphas:

            accuracies_alpha = []

            for beta in betas:

                # Get the data features for the classifier
                path_doc_topic_matrix = args.path_write_data + 'matrices/doc_topic_mat/doc_topic_mat_ntopic' + str(
                    args.n_topic_tune) + '_alpha' + str(alpha) + '_beta' + str(beta) + '.npz'

                # If the data doesn't exists yet, generate it
                if os.path.isfile(path_doc_topic_matrix):
                    doc_topic_matrix = scipy.sparse.load_npz(path_doc_topic_matrix)
                else:
                    path_model = args.path_write_data + 'models/tune/model_' + str(args.n_topic_tune) + '_alpha' + str(alpha) + '_beta' + str(beta)
                    doc_topic_matrix = get_doc_topic_matrices(path_model, df_balanced_data, n_docs=S.shape[0], vocabSize=len(id2word_final))

                    scipy.sparse.save_npz(path_doc_topic_matrix, doc_topic_matrix)

                accuracy = classifier(doc_topic_matrix, groundtruth, reg_coefs, list_train_idx, list_test_idx,
                                      list_val_idx)
                accuracies_alpha.append(accuracy)

            accuracies.append(accuracies_alpha)

        plot_accuracy_classifier_tune(accuracies, alphas, betas)

    else:

        print('Train classifiers on models with defined number of topics')

        # Create array of all topics for LDA
        n_topics_array = np.arange(args.min_n_topic, args.max_n_topic + 1, 5)
        accuracies = []

        for n_topics in n_topics_array:

            path_doc_topic_matrix = args.path_write_data + 'matrices/doc_topic_mat/doc_topic_mat_ntopic' + str(n_topics) + '.npz'

            if os.path.isfile(path_doc_topic_matrix):
                doc_topic_matrix = scipy.sparse.load_npz(path_doc_topic_matrix)
            else:
                path_model = args.path_write_data + 'models/model_' + str(n_topics)
                doc_topic_matrix = get_doc_topic_matrices(path_model, df_balanced_data, n_docs=S.shape[0], vocabSize=len(id2word_final))

                scipy.sparse.save_npz(path_doc_topic_matrix, doc_topic_matrix)

            accuracy = classifier(doc_topic_matrix, groundtruth, reg_coefs, list_train_idx, list_test_idx, list_val_idx)
            accuracies.append(accuracy)

        print(accuracies)
        plot_accuracy_classifier(accuracies, n_topics_array)

    print('Task done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #
    parser.add_argument('--path_dataset', dest='path_dataset',
                        default='/dlabdata1/youtube_large/yt_metadata_en.jsonl.gz',
                        help='path of the dataset')
    parser.add_argument('--path_write_data', dest='path_write_data',
                        default='/dlabdata1/youtube_large/olam/data/test/',
                        help='path to folder where we keep intermediate results')

    #
    parser.add_argument('--n_min_sub_classifier', dest='n_min_sub_classifier', type=int, default=None,
                        help='Threshold for the minimum number of subscrbers for relevant channels of videos in the '
                             'classification task')
    parser.add_argument('--n_min_views_classifier', dest='n_min_views_classifier', type=int, default=10000,
                        help='Threshold for the minimum number of views for relevant videos in the classification task')
    parser.add_argument('--n_top_vid_per_combination', dest='n_top_vid_per_combination', type=int, default=20,
                        help='Threshold for selecting the number videos with the most views for each combination of '
                             '\'category\', \'uploaded_year\' and \'channel_id\' for topic modelling. Must be the same when using it in data_processing.py')

    # Arguments for comparing classifier for model from min_n_topic topics to max_n_topic topics
    parser.add_argument('--min_n_topic', dest='min_n_topic', type=int, default=None,
                        help='The minimum number of topics for the LDA model. One classifier per model will be created and accuracies for each classifier are compared.')
    parser.add_argument('--max_n_topic', dest='max_n_topic', type=int, default=None,
                        help='The maximum number of topics for the LDA model. One classifier per model will be created and accuracies for each classifier are compared.')

    # Arguments for comparing classifier for model with n_topic_tune topics with specifics hyperparameters (docConcentration and topicConcentration)
    parser.add_argument('--tune', dest='tune', default=False, action='store_true',
                        help='Boolean that indicate that tuning the docConcentration and topicConcentration will be performed. Required n_topic_tune.')
    parser.add_argument('--n_topic_tune', dest='n_topic_tune', type=int, default=None,
                        help='The number of topics on which docConcentration and topicConcentration will be tuned.')

    #  Arguments related to Pyspark for transforming the data from the LDA model for the classifier
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=4,
                        help='The number of jobs for transforming the data for the classifier from the LDA model')
    parser.add_argument('--executor_mem', dest='executor_mem',
                        type=int, default=4, help='The memory in g for each executor')
    parser.add_argument('--driver_mem', dest='driver_mem',
                        type=int, default=64, help='The memory in g for the driver')

    args = parser.parse_args()

    main()

