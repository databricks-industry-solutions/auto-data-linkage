# Databricks notebook source
# MAGIC %pip install delta-sharing --quiet

# COMMAND ----------

username = spark.sql('select current_user() as user').collect()[0]['user']
username

# COMMAND ----------

configpath = f"/Workspace/Repos/{username}/splink-public-sector-hackathon/setup/config.share"

# COMMAND ----------

import delta_sharing

# COMMAND ----------

client = delta_sharing.SharingClient(configpath) 

client.list_all_tables()

# COMMAND ----------

def fetch_table(table_name):
    share_name = 'splink-share'
    schema_name= 'splink'

    df = spark.read.format("deltaSharing").load(f"file:{configpath}#{share_name}.{schema_name}.{table_name}")
    df.write.format("delta").save(f"dbfs:/Filestore/Users/{username}/{table_name}")
    df = spark.read.format("delta").load(f"dbfs:/Filestore/Users/{username}/{table_name}")
    return df

# COMMAND ----------

df_companies_with_officers = fetch_table("companies_with_officers")
df_companies_with_officers.display()

# COMMAND ----------

df_payment_practices_of_uk_buyers = fetch_table("payment_practices_of_uk_buyers")
df_payment_practices_of_uk_buyers.display()

# COMMAND ----------

df_company_officers = fetch_table("company_officers")
df_company_officers.display()

# COMMAND ----------

df.write.format('parquet').save("/mnt/landing/splinkdata")

# COMMAND ----------

df.write.format('parquet').save(f'/mnt/landing/{table_name}')

# COMMAND ----------


