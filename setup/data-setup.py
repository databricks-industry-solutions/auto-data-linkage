# Databricks notebook source
username = spark.sql('select current_user() as user').collect()[0]['user']


# COMMAND ----------

def fetch_table(table_name, reset):
    share_name = 'splink-share'
    schema_name= 'splink'
    
    if reset:
        mode='overwrite'
    else:
        mode='ignore'
    
    df = spark.read.format("delta").load(f"/mnt/source/splinkdata/delta/{table_name}")
    df.write.mode(mode).format("delta").save(f"dbfs:/Filestore/Users/{username}/{table_name}")
    df = spark.read.format("delta").load(f"dbfs:/Filestore/Users/{username}/{table_name}")
    return df

# COMMAND ----------

df_companies_with_officers = fetch_table("companies_with_officers", reset=False)
df_companies_with_officers.display()

# COMMAND ----------

df_payment_practices_of_uk_buyers = fetch_table("payment_practices_of_uk_buyers", reset=False)
df_payment_practices_of_uk_buyers.display()

# COMMAND ----------

df_company_officers = fetch_table("company_officers", reset=False)
df_company_officers.display()

# COMMAND ----------

display(dbutils.fs.ls(f"/Filestore/Users/{username}"))

# COMMAND ----------

def clear_user_data(table_name):
    dbutils.fs.rm(f"/Filestore/Users/{username}/{table_name}", True)

# COMMAND ----------

# clear_user_data("company_officers")
# clear_user_data("payment_practices_of_uk_buyers")
# clear_user_data("companies_with_officers")

# COMMAND ----------


