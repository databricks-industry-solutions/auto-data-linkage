# Databricks notebook source
# MAGIC %md
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/sample-repo. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sample-accelerator
import pandas as pd

## NEED TO CHANGE BRANCH
df = spark.createDataFrame(pd.read_csv("https://raw.githubusercontent.com/databricks-industry-solutions/auto-data-linkage/example_notebook_update/datasets/febrl1.csv"))

df.write.mode("overwrite").saveAsTable("febrl1")