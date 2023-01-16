import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
import pandas as pd

spark = SparkSession.builder.appName("Batch inference with MLflow DL inference pipeline").getOrCreate()

# load a logged model with run id and model name
# logged_model = 'runs:/37b5b4dd7bc04213a35db646520ec404/inference_pipeline_model'
# or load a registered model with a version number
#  model_uri=f"models:/{model_name}/{model_version}"

model_name = "inference_pipeline_model"
model_version = 1
logged_model = f'models:/{model_name}/{model_version}'

# Load model as a Spark UDF.
print(f"Loading model {logged_model}")
loaded_model = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=logged_model,
    result_type=StringType()
)
print("Finished loading model as Spark UDF")

# pyfunc_udf = mlflow.pyfunc.spark_udf(<path-to-model-with-signature>)
# df = spark_df.withColumn("prediction", pyfunc_udf())

# Predict on a Spark DataFrame.
df: pd.DataFrame = spark.read.csv('../data/imdb/test.csv', header=True)
df = df.select('review').withColumnRenamed('review', 'text')
df = df.withColumn('predictions', loaded_model())

df.show(n = 10, truncate=80, vertical=True)