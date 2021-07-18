# Pyspark demo

from pyspark.sql.functions import row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType, ArrayType
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

spark = SparkSession.builder.appName("SparkByExamples.com").getOrCreate()

print(spark)
rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 56])
rdd.count()

print("RDD COUNT " + str(rdd.count()))


data = [('James', '', 'Smith', '1991-04-01', 'M', 3000),
        ('Michael', 'Rose', '', '2000-05-19', 'M', 4000),
        ('Robert', '', 'Williams', '1978-09-05', 'M', 4000),
        ('Maria', 'Anne', 'Jones', '1967-12-01', 'F', 4000),
        ('Jen', 'Mary', 'Brown', '1980-02-17', 'F', -1)
        ]

columns = ["firstname", "middlename", "lastname", "dob", "gender", "salary"]
df = spark.createDataFrame(data=data, schema=columns)

df.show()


# withColumn() + cast()
df.withColumn("salary", sf.col("salary").cast("Integer")).show()


# withColumn() - update
df.withColumn("salary", sf.col("salary")*100).show()


# withColumn() - create
df.withColumn("CopiedColumn", sf.col("salary")*-1).show()


# withColumn() - new col
df.withColumn("Country", sf.lit("USA")) \
  .withColumn("anotherColumn", sf.lit("anotherValue")) \
  .show()


# withColumnRenamed()
df.withColumnRenamed("gender", "sex").show(truncate=False)


# drop
df.drop("salary").show()
df.drop("dob", "salary").show()

### Structypes +  Filter

data = [
    (("James", "", "Smith"), ["Java", "Scala", "C++"], "OH", "M"),
    (("Anna", "Rose", ""), ["Spark", "Java", "C++"], "NY", "F"),
    (("Julia", "", "Williams"), ["CSharp", "VB"], "OH", "F"),
    (("Maria", "Anne", "Jones"), ["CSharp", "VB"], "NY", "M"),
    (("Jen", "Mary", "Brown"), ["CSharp", "VB"], "NY", "M"),
    (("Mike", "Mary", "Williams"), ["Python", "VB"], "OH", "M")
]

data

schema = StructType([
    StructField('name', StructType([
        StructField('firstname', StringType(), True),
        StructField('middlename', StringType(), True),
        StructField('lastname', StringType(), True)
    ])),
    StructField('languages', ArrayType(StringType()), True),
    StructField('state', StringType(), True),
    StructField('gender', StringType(), True)
])

df = spark.createDataFrame(data=data, schema=schema)
df.printSchema()
df.show(truncate=False)

# filter
df.filter(df.state == "OH").show(truncate=False)
df.filter(sf.col("state") == "OH").show(truncate=False)
df.filter((df.state == "OH") & (df.gender == "M")).show(truncate=False)


# filter based on list
li = ["OH", "CA", "DE"]
df.filter(df.state.isin(li)).show()
df.filter(~df.state.isin(li)).show()


# filter startswith, endswith, contains
df.filter(df.state.startswith("N")).show()
df.filter(df.state.endswith("H")).show()
df.filter(df.state.contains("H")).show()


# filter on Array Column
df.filter(sf.array_contains(df.languages, "Java")).show(truncate=False)


# filter nested struc column
df.filter(df.name.lastname == "Williams").show(truncate=False)


# Aggregate Functions

simpleData = [("James", "Sales", 3000),
              ("Michael", "Sales", 4600),
              ("Robert", "Sales", 4100),
              ("Maria", "Finance", 3000),
              ("James", "Sales", 3000),
              ("Scott", "Finance", 3300),
              ("Jen", "Finance", 3900),
              ("Jeff", "Marketing", 3000),
              ("Kumar", "Marketing", 2000),
              ("Saif", "Sales", 4100)
              ]
schema = ["employee_name", "department", "salary"]
df = spark.createDataFrame(data=simpleData, schema=schema)
df.printSchema()
df.show(truncate=False)


# approx_count_distinct()
df.select(sf.approx_count_distinct("salary")).collect()[0][0]


# avg
str(df.select(sf.avg("salary")).collect()[0][0])


# collect list
a = df.select(sf.collect_list("salary"))


# collect set
df.select(sf.collect_set("salary")).show(truncate=False)


# count distinct
df2 = df.select(sf.countDistinct("department", "salary"))
df2.show(truncate=False)
print("Distinct Count of Department & Salary: "+str(df2.collect()[0][0]))


# count
df.select(sf.count("salary")).collect()[0][0]


# first
df.select(sf.first("salary")).show(truncate=False)


# Groupby
df.groupBy("department").count().show()
df.groupBy("department").max("salary").show()

# si no uso sf.sum no corre. sum a secas es otra funcion y rompe
df.groupBy("department")\
    .agg(sf.sum("salary").alias("sum_salary"),
         sf.avg("salary").alias("avg_salary")
         )\
    .filter(sf.col("sum_salary") > 10000)\
    .show(truncate=False)


# Window Functions
df.show(truncate=False)


# generas partición sobre la cual aplicas las funciones
windowSpec = Window.partitionBy("department").orderBy("salary")

# row number
df.withColumn("row_number", sf.row_number().over(
    windowSpec)).show(truncate=False)


# rank
df.withColumn("rank", sf.rank().over(windowSpec)).show(truncate=False)
df.withColumn("dense_rank", sf.dense_rank().over(windowSpec)) \
    .show()


# lag
df.withColumn("lag", sf.lag("salary", 2).over(windowSpec)) \
    .show()


# Agg en Windows function
windowSpecAgg = Window.partitionBy("department")

df.withColumn("row", sf.row_number().over(windowSpec))\
  .withColumn("sum", sf.sum("salary").over(windowSpecAgg))\
  .filter(sf.col("row") == 1)\
    .show()


# Pruebas
a1 = df.groupBy('firstname', 'gender').agg(sf.countDistinct('salary'))
a1.show()
a2 = a1.filter(~(sf.col("firstname") == "Jen")).select(
    'firstname', 'gender').collect()
#a2 = a1.select('firstname','gender').collect()
a3 = [(row.firstname, row.gender) for row in a2]
a3_str = [",".join([str(x) for x in item]) for item in a3]
a3
df.withColumn("combined_id", sf.concat(sf.col("firstname"), sf.lit(","), sf.col("gender")))\
    .filter(sf.col("combined_id").isin(a3_str))\
    .show()
