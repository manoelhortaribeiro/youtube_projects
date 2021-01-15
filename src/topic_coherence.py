import argparse
import os
import scipy.sparse

from utils import *


def main():

    if args.tune_alpha_beta and args.n_topic is None:
        raise ValueError('The number of topics need to be provided when tuning the hyperparemeters alpha and beta')

    print('Loading data...')
    S_final = scipy.sparse.load_npz(args.path_data + 'matrices/S_final.npz')
    id2word = load_pickle(args.path_data + 'id2word_final_old.pickle')

    # Load texts and corpus if they already exists, otherwise compute and save them
    if not os.path.isfile(args.path_data + 'texts.pickle'):
        texts = get_texts(S_final, id2word)
        save_to_pickle(texts, 'texts', args.path_data)
    else:
        texts = load_pickle(args.path_data + 'texts.pickle')

    if not os.path.isfile(args.path_data + 'corpus.pickle'):
        corpus = get_corpus(S_final)
        save_to_pickle(corpus, 'corpus', args.path_data)
    else:
        corpus = load_pickle(args.path_data + 'corpus.pickle')

    spark, conf = create_spark_session(n_jobs=1, executor_mem=4, driver_mem=16)

    if args.tune_alpha_beta:
        alphas = [0.1, 0.5, 0.9, 0.95, 1]
        betas = [0.01, 0.05, 0.1, 0.5]

        print('Computing coherence scores for models with specific hyperparameters...')
        coherence_scores_cv, coherence_scores_umass = get_coherence_scores_tune(S_final,
                                                                                spark,
                                                                                args.path_data,
                                                                                args.n_topics,
                                                                                alphas,
                                                                                betas,
                                                                                texts,
                                                                                corpus,
                                                                                id2word,
                                                                                args.n_jobs)
        coherence_plot_tune(args.n_topics, coherence_scores_cv, coherence_scores_umass, alphas, betas)
    else:
        print('Computing coherence scores for LDA models with some number k of topics...')
        coherence_scores_cv, coherence_scores_umass, n_topics_list = get_coherence_scores(S_final,
                                                                                          spark,
                                                                                          args.path_data,
                                                                                          args.min_n_topic,
                                                                                          args.max_n_topic,
                                                                                          texts,
                                                                                          corpus,
                                                                                          id2word,
                                                                                          args.n_jobs)
        coherence_plot(coherence_scores_cv, coherence_scores_umass, n_topics_list)

    print('Task done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments for comparing coherence scores between number of topics
    parser.add_argument('--min_n_topic', dest='min_n_topic', type=int, default=None,
                        help='The number of topics from which the topic coherences will be computed')
    parser.add_argument('--max_n_topic', dest='max_n_topic', type=int, default=None,
                        help='The number of topics until which the topic coherences will be computed')

    # Arguments for comparing coherence scores between different alphas and betas
    parser.add_argument('--n_topics', dest='n_topics', type=int, default=None,
                        help='Number of topics when `tune_alpha_beta` is True. Required `tune_alpha_beta` to be True.')
    parser.add_argument('--tune_alpha_beta', dest='tune_alpha_beta', default=False, action='store_true',
                        help='If True, tune the hyperparameters for alpha (docConcentration) and beta '
                             '(topicConcentration) for a given number of topic. Required `n_topic` not None.')

    # General arguments
    parser.add_argument('--path_data', dest='path_data',
                        default='/dlabdata1/youtube_large/olam/data/test/',
                        help='path to folder where we keep intermediate results')
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=1, help='Number of cores for parallelization')

    args = parser.parse_args()

    main()
