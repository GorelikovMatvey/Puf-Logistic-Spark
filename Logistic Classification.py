import pyspark
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Инициализация SparkSession
spark = SparkSession.builder \
    .appName("PUF Logistic Regression") \
    .master("local[*]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.instances", "1") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()

# Загрузка данных
train_path = "data/train_5xor_128dim.csv"
test_path = "data/test_5xor_128dim.csv"
train_df = spark.read.csv(train_path, header=False, inferSchema=True)
test_df = spark.read.csv(test_path, header=False, inferSchema=True)

# Список признаков
feature_columns = [f"_c{i}" for i in range(128)]
label_column = "_c128"

# Удаление пустых значений
train_df = train_df.na.drop()
test_df = test_df.na.drop()

# Замена -1 на 0
train_df = train_df.replace(-1.0, 0)
test_df = test_df.replace(-1.0, 0)

# Формирование вектора признаков
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# Обучение модели
lr = LogisticRegression(featuresCol="features", labelCol=label_column)
start_time = time.time()
model = lr.fit(train_df)
print("\nВремя обучения: {:.2f} сек".format(time.time() - start_time))

# Прогноз и оценка
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol=label_column, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("\nТочность модели: {:.4f}".format(accuracy))

# Остановка Spark
spark.stop()
