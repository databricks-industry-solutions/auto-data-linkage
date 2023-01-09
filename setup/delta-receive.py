# Databricks notebook source
# MAGIC %pip install delta-sharing

# COMMAND ----------

configpath = "/Workspace/Repos/ref/splink-public-sector-hackathon/setup/config.share"

# COMMAND ----------

import delta_sharing

# COMMAND ----------

client = delta_sharing.SharingClient(configpath) 

client.list_all_tables()

# COMMAND ----------

share_name = 'splink-share'
schema_name= 'splink'
table_name = 'companies_with_officers'


#df = spark.read.format("deltaSharing").load(f"{profile_path}#{share_name}.{schema_name}.{table_name}")
df = spark.read.format("deltaSharing").load(f"file:{configpath}#{share_name}.{schema_name}.{table_name}")

#.limit(10)

# COMMAND ----------

df2 = spark.read.format("deltaSharing").load(f"{profile_path}#{share_name}.{schema_name}.company_officers")
#.limit(10)

# COMMAND ----------

display(df)

# COMMAND ----------

dbutils.fs.mounts()

# COMMAND ----------

df.write.format('parquet').save("/mnt/landing/splinkdata")

# COMMAND ----------

df.write.format('parquet').save(f'/mnt/landing/{table_name}')

# COMMAND ----------


