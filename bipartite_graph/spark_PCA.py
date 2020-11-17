import json
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.feature import PCA



def main():

    # create the session
    spark = SparkSession.builder.appName('test for bipartite graph').getOrCreate()

    # Load data
    print('Loading data...')
    df_load = spark.read.json('/user/jouven/pyspark_data.json')

    # Process data
    data_process = []

    print('Processing data...')
    for row in df_load.collect():
        features = row['features']
        data_process.append([row['channel_index'], SparseVector(
            features['size'], features['indices'], features['values'])])

    data_process = spark.createDataFrame(
        data_process, ['channel_index', 'features'])

    pca = PCA(k=100, inputCol="features", outputCol="pca_features")
    model = pca.fit(data_process)
    model.transform(data_process)

    print('Task done!')


if __name__ == "__main__":
    main()