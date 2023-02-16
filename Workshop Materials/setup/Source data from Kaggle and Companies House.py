# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### This notebook will create the dataset used in the workshop. 
# MAGIC 
# MAGIC In the workshop we used a dataset of corporate officers scraped from Companies House. The companies these individuals are officers off are sourced from Kaggle, from the dataset [**Payment Practices of UK Buyers**](https://www.kaggle.com/datasets/saikiran0684/payment-practices-of-uk-buyers). This dataset is made available by the UK Government under the Open Government Licence, and provides details about the payment practices of large companies. 
# MAGIC 
# MAGIC The company number provided as part of the Payment Practices dataset is used to retrive the corporate officers of that company from the Companies House API.
# MAGIC 
# MAGIC This notebook requires you to have an account with Kaggle and and a Kaggle API key. Details on how to do this are [here](https://github.com/Kaggle/kaggle-api).
# MAGIC This notebook requires you to have an account with Companies House and and a Companies House API key. Follow this [guide](https://www.pythonontoast.com/python-data-interface-companies-house/) in order to register with Companies House and generate an API key.
# MAGIC 
# MAGIC All data downloaded is written to DBFS in the path "/Users/*username*/...", and made available as Delta Tables in a database named *username*_splink_data
# MAGIC 
# MAGIC PS - creating the full extract takes about 7 hours to run.
# MAGIC 
# MAGIC PPS - tested on DBR 11.3 LTS

# COMMAND ----------

pip install kaggle

# COMMAND ----------

import os

# COMMAND ----------

# MAGIC %md
# MAGIC This creates the location for where the Kaggle data will be downloaded too.

# COMMAND ----------

username = spark.sql('select current_user() as user').collect()[0]['user']
os.environ["USERNAME"]=username
download_path = f"dbfs:/Users/{username}/datasets/kaggle/payment-practices-of-uk-buyers"
dbutils.fs.mkdirs(download_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This will download and unzip the data from kaggle. Replace KAGGLE_USERNAME and KAGGLE_API_KEY with your kaggle credentials.

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC mkdir ~/.kaggle
# MAGIC echo '{"username":<KAGGLE_USERNAME>,"key":<KAGGLE_API_KEY>}' > ~/.kaggle/kaggle.json
# MAGIC kaggle datasets download --force -d saikiran0684/payment-practices-of-uk-buyers -p /dbfs/Users/$USERNAME/datasets/kaggle/payment-practices-of-uk-buyers
# MAGIC rm ~/.kaggle/kaggle.json
# MAGIC cd /dbfs/Users/$USERNAME/datasets/kaggle/payment-practices-of-uk-buyers
# MAGIC unzip payment-practices-of-uk-buyers.zip

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we read the downloaded Kaggle data and write it out as a Delta Table to our Splink database.

# COMMAND ----------

database = f"{username.replace('.', '_').replace('@', '_')}_splink_data"
spark.sql(f"create database if not exists {database}")
df=(
spark.read.format('csv').option("header", True)
.load(f"dbfs:/Users/{username}/datasets/kaggle/payment-practices-of-uk-buyers/payment-practices.csv")
)


# need to rename columns to get rid of spaces and percentanges
new_columns = [x.replace(" ", "_").replace("%", "percentage").replace("-", "_").replace("(", "").replace(")", "") for x in df.columns]
new_df=df.toDF(*new_columns)
(new_df
.write
.saveAsTable(f"{database}.payment_practices_of_uk_buyers")
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The next step is to retrive the Corporate Officers information from Companies House. This API is rate limited to 500 calls per minute, so we must include that in our retrieval logic.

# COMMAND ----------

import requests
import time
from pyspark.sql import functions as F
from pyspark.sql import Window
import pandas as pd 
from pyspark.sql import types
from pyspark.sql.functions import collect_set, size


# COMMAND ----------

# MAGIC %md
# MAGIC These functions simplify calling the API and parsing the response. 
# MAGIC Replace the <'auth key'> value with your Companies House API key.

# COMMAND ----------

key = <'auth key'>
def get_officers_api(companyNumber):
  api=f"https://api.company-information.service.gov.uk/company/{companyNumber}/officers"
  response = requests.get(api
               , headers={'Authorization': key}

              ).json()
  return response

def response_parser(response):
  items = response['items']
  output =[]
  for item in items:
    record = {}
    address = item.get('address', {})
    record['address_line_1'] = address.get('address_line_1')
    record['locality'] = address.get('locality')
    record['country'] = address.get('country')
    record['premises'] = address.get('premises')
    record['region'] = address.get('region')
    record['postal_code'] = address.get('postal_code')
    record['officer_role'] = item.get('officer_role')
    record['name'] = item.get('name')
    record['country_of_residence'] = item.get('country_of_residence')
    record['occupation'] = item.get('occupation')
    record['nationality'] = item.get('nationality')
    dob = item.get('date_of_birth')
    if dob:
      if dob.get('month') < 10:
        month = "0"+str(dob.get('month'))
      else:
        month = str(dob.get('month'))
      record['date_of_birth'] = str(dob.get('year')) + "_" + month
    else:
      record['date_of_birth'] = None
    record['officer_role'] = item.get('officer_role')
    output.append(record)

  return output

def get_directors(companyNumber, Report_Id):
  response=get_officers_api(companyNumber)
  parsed = response_parser(response)
  for x in parsed:  
    x["company_number"] = companyNumber
    x["Report_Id"] = Report_Id
  return parsed


# COMMAND ----------

# MAGIC %md 
# MAGIC Now we just need to extract the company numbers and get our data! We will make 500 requests, and then wait until enough time has passed to make another batch of requests, erring on the side of caution by waiting a bit longer than necessary. This creates a database table containing the information. 
# MAGIC 
# MAGIC NB - this takes between 7-9 hours to complete.

# COMMAND ----------

data_to_link = spark.read.table(f"{database}.payment_practices_of_uk_buyers")
data_to_enrich = (
  data_to_link
  .select("Report_Id", "company_number")
  .withColumn("row_num", F.row_number().over(Window().partitionBy().orderBy(F.lit(True))))
).cache()
n_records = data_to_enrich.count()

increment=500
i=500
enriched_results = []
while i< (n_records + increment):
  local_data = data_to_enrich.filter(
     (F.col("row_num") > (i-increment)) &
     (F.col("row_num") <= (i))
  ).toPandas()
  for row in local_data.iterrows():
    _ = row[1]
    try:
      enriched_results.append(
        get_directors(_.company_number, _.Report_Id)
      )
    except:
      pass
  
  
  spark.createDataFrame(
    pd.concat([pd.DataFrame().from_dict(x) for x in enriched_results ])
    ).write.mode("append").saveAsTable(f"{database}.company_officers")
  print(f"enriched {len(enriched_results)} records and saved")
  print(f"{spark.sql(f'select count(distinct report_id) as num_records_enriched, count(*) as num_officers from {database}.company_officers').collect()} records enriched so far")
  i+= increment 
  enriched_results = []
  time.sleep(60*6)
  



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the purposes of the workshop, we're going to focus on just the personal information

# COMMAND ----------

username = spark.sql('select current_user() as user').collect()[0]['user']
database = f"{username.replace('.', '_').replace('@', '_')}_splink_data"

from pyspark.sql import functions as F
cleansed_company_officers = (
  spark.read.table(f"{database}.company_officers")
  .withColumn("forenames", F.split("name", ",")[1])
  .withColumn("surname", F.split("name", ",")[0])
  .select(
    "name"
    ,"country_of_residence"
    ,"nationality"
    ,"date_of_birth"
    ,"forenames"
    ,"surname"
    ,"postal_code"
    ,"premises"
    ,"address_line_1"
    ,"locality"
    ,"region"
  )
.drop_duplicates()
  .withColumn("uid", F.monotonically_increasing_id())
)

cleansed_company_officers.write.mode("append").saveAsTable(f"{database}.cleansed_company_officers")

display(cleansed_company_officers)
