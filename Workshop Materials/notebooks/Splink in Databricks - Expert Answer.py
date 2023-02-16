# Databricks notebook source
# MAGIC %md
# MAGIC # Splink on Databricks - an "expert" solution
# MAGIC Not the only (or optimal) solution, but one based on how the author would typically approach the problem.

# COMMAND ----------

# MAGIC %pip install splink --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Libraries
# MAGIC 
# MAGIC **Note the addition of the comparison _level_ library to help build custom comparisons with multiple different levels.**

# COMMAND ----------

from splink.spark.spark_linker import SparkLinker
from splink.databricks.enable_splink import enable_splink
import splink.spark.spark_comparison_library as cl
import splink.spark.spark_comparison_level_library as cll


import pandas as pd

from IPython.display import IFrame

from utils.splink_linker_model import SplinkLinkerModel
from utils.mlflow_utils import *
from utils.mlflow import splink_mlflow

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC Set parameters based on active user

# COMMAND ----------

username = spark.sql('select current_user() as user').collect()[0]['user']
db_name = f"{username.replace('.', '_').replace('@', '_')}_splink_data"
db_name

# COMMAND ----------

table_name = "cleansed_company_officers"
data = spark.read.table(db_name+'.'+table_name)

# COMMAND ----------

# DBTITLE 1,Remove cached Splink tables
#Splink has an issue where changes to the input data are not reflected in it's cached tables. Run this step after doing any feature engineering.
x = spark.sql(f"show tables from {db_name} like '*__splink__*'").collect()
for _ in x:
    spark.sql(f"drop table {db_name}.{_.tableName}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Our Turn
# MAGIC 
# MAGIC We now have a baseline model which no doubt could use some improvement. We approach this as follows:
# MAGIC 
# MAGIC 1. **Nodes** (input records) - Explore the data with splink's `missingness` and `profile_columns` methods - this is an iterative process to identify transformations to make to the input data:
# MAGIC     - Data cleaning/standardisation - making sure all values in a column are formatted the same way, handling nulls or anomalies
# MAGIC     - Feature engineering - creating new derived columns that could be useful for blocking or making custom comparisons
# MAGIC 2. **Model training** - Having developed an understanding of the data:
# MAGIC     - Refine blocking rules for training and predictions
# MAGIC     - Assemble comparison columns (one for each identifier or attribute), with multiple comparison levels covering potential observations in the data (distinct ways matching records _might_ differ)
# MAGIC     - Check model is self-consistent using `linker.parameter_estimate_comparisons_chart()`
# MAGIC     - Check model makes intuitive sense using `linker.match_weights_chart()`
# MAGIC 3. **Edges** - `linker.predict()` generates match scores between pairs of nodes (i.e. "edges"):
# MAGIC     - How are the match scores distributed? Expect a bimodal distribution (definite matches + definite non-matches)
# MAGIC     - If there are ambiguous scores (0.1 < `match_proability` < 0.9; `match_weight` close to zero) we can use the waterfall chart to investigate
# MAGIC 6. **Clusters** - once we're happy with our model, we probably have a rough idea of what threshold match score to use to define a match (e.g. >0.9). Spot checking a sample of clusters with `cluster_studio_dashboard` we can check for a couple of things, the biggest risk being due to false positives (linking records that aren't genuine matches):
# MAGIC     - Are there any "hairball" clusters that have become overly large because several genuine clusters have been merged?
# MAGIC     - Ideal clusters will be fully-connected (every node is directly linked to every other) - are there clusters that appear to be two or more subclusters only connected by a single (suspicious) edge?

# COMMAND ----------

data.toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Nodes
# MAGIC > **Explore the data with splink's `missingness` and `profile_columns` methods - this is an iterative process to identify transformations to make to the input data:**
# MAGIC > - **Data cleaning/standardisation - making sure all values in a column are formatted the same way, handling nulls or anomalies**
# MAGIC > - **Feature engineering - creating new derived columns that could be useful for blocking or making custom comparisons**
# MAGIC 
# MAGIC 
# MAGIC Initial ideas for standardisation:
# MAGIC - Clean up `country_of_residence` and `nationality`
# MAGIC - Create `year_of_birth` for blocking/fuzzy comparisons
# MAGIC - Split forenames into `forename1`, `forename2` etc.
# MAGIC - Create `full_address` to include `premises` + `address_line_1` + `locality` + `region` + `postal_code`
# MAGIC - Set all string columns to upper case - string comparator functions are typically case-sensitive
# MAGIC - Create partial postcode columns (`area`, `district`, `sector`) - see image [here](https://user-images.githubusercontent.com/7570107/136946496-8769b06c-e4a6-488d-95f1-526946d96aa7.png)

# COMMAND ----------

import pyspark.sql.functions as f

# STANDARDISED DATA

# Names
data_std = data\
    .withColumn("forename1", f.split(f.upper(data['forenames']), ' ').getItem(1))\
    .withColumn("forename2", f.split(f.upper(data['forenames']), ' ').getItem(2))\
    .withColumn("forename3", f.split(f.upper(data['forenames']), ' ').getItem(3))\
    .withColumn("forename4", f.split(f.upper(data['forenames']), ' ').getItem(4))\
    .withColumn("names_arr", f.expr("array_sort(array(forename1, forename2, surname))"))

# DOB
data_std = data_std.withColumn("year_of_birth", f.expr("CAST(left(date_of_birth, 4) as int)"))

# Postcodes
data_std = data_std\
    .withColumn("postcode_validated", f.expr("CASE WHEN postal_code RLIKE '^[A-Z]{1,2}\\\\d{1,2}[A-Z]?\\\\s?\\\\d[A-Z]{2}' THEN postal_code ELSE NULL END"))\
    .withColumn("pc_sector", f.expr("REGEXP_EXTRACT(postcode_validated, '^[A-Z]{1,2}\\\\d{1,2}[A-Z]?\\\\s?\\\\d', 0)"))\
    .withColumn("pc_district", f.expr("REGEXP_EXTRACT(postcode_validated, '^[A-Z]{1,2}\\\\d{1,2}', 0)"))\
    .withColumn("pc_area", f.expr("REGEXP_EXTRACT(postcode_validated, '^[A-Z]{1,2}', 0)"))\
    .withColumn("full_address", f.expr("CONCAT_WS(' ', premises, address_line_1, locality, region, postal_code)"))

# Country
data_std = data_std.withColumn(
    "country_clean", 
    f.expr("""CASE 
    WHEN country_of_residence IN ('England', 'Scotland', 'Wales', 'Northern Ireland', 'Great Britain') THEN 'United Kingdom'
    WHEN regexp_like(country_of_residence, '^(Uk$|Uk\\\\s|U.+ingdom)') THEN 'United Kingdom'
    WHEN regexp_like(country_of_residence, 'shire$') THEN 'United Kingdom'
    WHEN regexp_like(country_of_residence, '^(Usa|United States)') THEN 'United States'
    ELSE country_of_residence END"""))

# Nationality
data_std = data_std.withColumn(
    "nationality_clean", 
    f.expr("""CASE 
    WHEN nationality IN ('English', 'Scottish', 'Welsh', 'Northern Irish', 'United Kingdom') THEN 'British'
    WHEN regexp_like(nationality, 'British|Gb$|Uk$') THEN 'British'
    WHEN nationality IN ('Usa', 'United States', 'Us Citizen', 'Us') THEN 'American'
    WHEN nationality = 'Other' THEN NULL
    ELSE nationality END"""))


# BUG WORKAROUND - run before creating a new linker to make sure it updates properly
spark.sql("DROP TABLE IF EXISTS __splink__df_concat_with_tf")

linker = SparkLinker(data_std, spark=spark)

linker.profile_columns("postcode_validated", top_n=30, bottom_n=10)

# COMMAND ----------

# MAGIC %md
# MAGIC Some of these changes may be overkill, and others could be more extensive if necessary.
# MAGIC 
# MAGIC `nationality_clean` for example has reduced the number of distinct values from 652 to 482, helping maximise the chance of finding a match on nationality.
# MAGIC 
# MAGIC For the `forename` values however, we can see from the missingness chart below that `forename3` and `forename4` are rarely populated, so we probably won't want to use these new columns.

# COMMAND ----------

linker.missingness_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model training
# MAGIC >**Having developed an understanding of the data:**
# MAGIC >- **Refine blocking rules for training and predictions**
# MAGIC >- **Assemble comparison columns (one for each identifier or attribute), with multiple comparison levels covering potential observations in the data (distinct ways matching records _might_ differ)**
# MAGIC >- **Check model is self-consistent using `linker.parameter_estimate_comparisons_chart()`**
# MAGIC >- **Check model makes intuitive sense using `linker.match_weights_chart()`**
# MAGIC 
# MAGIC Broadly, the data contains just 4 attributes for each person:
# MAGIC 1. Name
# MAGIC 2. Address (incl. `country_of_residence`?)
# MAGIC 3. DOB
# MAGIC 4. Nationality

# COMMAND ----------

# Let's relax these (seeing as the dataset is relatively small),
# or simplify them using some of our new derived columns
blocking_rules_to_generate_predictions = [
  "l.names_arr = r.names_arr",
  "l.forename1 = r.forename1 and l.date_of_birth = r.date_of_birth",
  "l.full_address = r.full_address",
  "l.year_of_birth = r.year_of_birth and l.postal_code = r.postal_code",
]

# These are only used to roughly estimate the lambda parameter (probablity of two random records matching)
# No need to change as they have little impact on the relative match weights
deterministic_rules_to_estimate_lambda = [
  "l.name = r.name and levenshtein(r.date_of_birth, l.date_of_birth) <= 1",
  "l.address_line_1 = r.address_line_1 and levenshtein(l.name, r.name) <= 5",
  "l.name = r.name and levenshtein(l.address_line_1, r.address_line_1) <= 5",
]

# We want our training rules to be completely independent
# 1 - First we block on full name to estimate the parameters for all other columns
# 2 - Then we block on something else (address) so we can estimate the parameters for full name and DOB
# 3 - Then we can try a third model to double check all parameters
blocking_rules_for_model_training = [
  "l.name = r.name",
  "l.full_address = r.full_address",
  "l.date_of_birth = r.date_of_birth and l.nationality_clean = r.nationality_clean"
]

# COMMAND ----------

# MAGIC %md
# MAGIC The list of comparisons is where we define the complexity of the (Fellegi-Sunter) linkage model that we are fitting. Everything else helps us estimate the best possible parameters for our chosen model, but this is where we make sure the model is sufficiently sophisticated.
# MAGIC 
# MAGIC Some of our comparisons will be off-the-shelf functions from Splink, others can be customised to varying degrees as shown.

# COMMAND ----------

# Name columns are correlated so ideally we combine them into one comparison. 
# Here we check for a few possible observations:
# - All names match
# - Names are jumbled up [REMOVED AS NEVER OBSERVED]
# - One or more names match (but not necessarily in the same column)
# - One or more name columns is a fuzzy match
# - Else, no names match
name_comparison = {
    "output_column_name": "name",
    "comparison_levels": [
        cll.null_level("name"),
        cll.exact_match_level("name"),
        cll.jaro_winkler_level("name", 0.95),
#         {
#             "sql_condition": "names_arr_l = names_arr_r",
#             "label_for_charts": "Exact names (shuffled)"
#         },
        {
            "sql_condition": "arrays_overlap(names_arr_l, names_arr_r)",
            "label_for_charts": "Any names match"
        },
        {
            "sql_condition": """
                (jaro_winkler(forename1_l, forename1_r) > 0.9) 
                OR (jaro_winkler(forename2_l, forename2_r) > 0.9) 
                OR (jaro_winkler(surname_l, surname_r) > 0.9)
                """,
            "label_for_charts": "Jaro-Winkler any name"
        },
        cll.else_level()
    ]
}

# Similarly, we treat address as a single attribute rather than several correlated ones
# If the full address doesn't match, we check the postcode, then larger geographical areas
# Some of these will be considerably more common than others so we turn on TF adjustments
address_comparison = {
    "output_column_name": "address",
    "comparison_levels": [
        cll.null_level("full_address"),
        cll.exact_match_level("full_address"),
        cll.jaro_winkler_level("full_address", 0.95),
        cll.exact_match_level("postcode_validated", term_frequency_adjustments=True, include_colname_in_charts_label=True),
        cll.exact_match_level("pc_sector", term_frequency_adjustments=True, include_colname_in_charts_label=True),
        cll.exact_match_level("pc_district", term_frequency_adjustments=True, include_colname_in_charts_label=True),
        cll.exact_match_level("pc_area", term_frequency_adjustments=True, include_colname_in_charts_label=True),
        cll.exact_match_level("country_clean", term_frequency_adjustments=True, include_colname_in_charts_label=True),
        cll.else_level()
    ]
}

# DOB is distributed pretty uniformly so no need for TF adjustments
dob_comparison = {
    "output_column_name": "dob",
    "comparison_levels": [
        cll.null_level("date_of_birth"),
        cll.exact_match_level("date_of_birth", include_colname_in_charts_label=True),
        cll.exact_match_level("year_of_birth", include_colname_in_charts_label=True),
        cll.else_level()
    ]
}

comparisons = [
    name_comparison,
    address_comparison,
    dob_comparison,
    #cl.exact_match("country_clean", term_frequency_adjustments=True),
    cl.exact_match("nationality_clean", term_frequency_adjustments=True)
]

# COMMAND ----------

settings = {
  "retain_intermediate_calculation_columns": True,
  "retain_matching_columns": True,
  "link_type": "dedupe_only",
  "unique_id_column_name": "uid",
  "comparisons": comparisons,
  "em_convergence": 0.01,
  "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions
}

linker.initialise_settings(settings)

# Estimate lambda
linker.estimate_probability_two_random_records_match(deterministic_rules_to_estimate_lambda, recall=0.8)

# Estimate u-probabilities
linker.estimate_u_using_random_sampling(target_rows=1e7)

# Estimate m-probabilities
for training_rule in blocking_rules_for_model_training:
    linker.estimate_parameters_using_expectation_maximisation(training_rule)

# COMMAND ----------

# We have estimate the m probabilities 2 or 3 times with different blocking rules
# In theory, these should be the same - if they're VERY different, it's a sign something is wrong
linker.parameter_estimate_comparisons_chart(include_u=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Our m-probability estimates from the different training models are broadly in agreement, except for the `name` comparison - this is likely due to some correlation between a match on name and the blocking rules.
# MAGIC 
# MAGIC The final model will take the harmonic mean of each of these parameters, so should settle on a reasonable final value even where there are these inconsistencies. The `match_weights_chart` will help us check our model seems sensible by telling us how much each comparison level adjusts our evidence for or against a match:
# MAGIC - The "prior match weight" is the starting point for any record comparison - if we know nothing else about these records, there is a very low probability of them being a match
# MAGIC - For each comparison level, the match weight can be positive (green), negative (red) and the larger the match weight, the more important it is for establishing a match. For example:
# MAGIC   - An exact match on name is very strong evidence for a match ✅
# MAGIC   - A match on postcode area is less significant evidence for a match ✅
# MAGIC   - No match on DOB is evidence against a match ❌
# MAGIC   - A mismatch on nationality is very strong evidence against a match ❌

# COMMAND ----------

linker.match_weights_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Edges
# MAGIC >**`linker.predict()` generates match scores between pairs of nodes (i.e. "edges"):**
# MAGIC > - **How are the match scores distributed? Expect a bimodal distribution (definite matches + definite non-matches)**
# MAGIC > - **If there are ambiguous scores** (0.1 < `match_probability` < 0.8 ; `match_weight` close to zero) **we can use the waterfall chart to investigate**
# MAGIC 
# MAGIC Before we run the prediction, we can check that our blocking rules aren't going to produce an impractical number of edges. For 150k records, we probably wouldn't expect to produce much more than 1 million edges. If we are, then our blocking rules are probably too lenient and we can change them before wasting a lot of computing resource.

# COMMAND ----------

linker.cumulative_num_comparisons_from_blocking_rules_chart()

# COMMAND ----------

# Make predictions
predictions = linker.predict()

# Relatively small dataset means this distribution will be less clear, but most match weights will be low, because most comparisons aren't matches
linker.match_weights_histogram(predictions)

# COMMAND ----------

path = f"/dbfs/Users/{username}/comparison_viewer.html"
linker.comparison_viewer_dashboard(
    predictions, 
    out_path=path, 
    overwrite=True
)

with open("/dbfs" + path, "r") as f:
    html=f.read()    
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC Using the comparison viewer above you can brush the bottom chart to select comparison levels in between the red and green ends to filter out the clear matches and non-matches. Clicking a bar will show an example waterfall chart to demonstrate how the match weight is calculated.
# MAGIC 
# MAGIC Another way to do this is shown below:
# MAGIC - Filter predictions according to their match probabilty
# MAGIC - Take a sample and output to dict
# MAGIC - Create a waterfall chart from these ambiguous edges

# COMMAND ----------

ambiguous_edges = predictions.as_spark_dataframe()\
    .filter("match_probability > 0.1 and match_probability < 0.8").limit(100)\
    .toPandas().to_dict(orient="records")

linker.waterfall_chart(ambiguous_edges, filter_nulls=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Using the slider at the bottom of the waterfall chart, we can quickly see that most of these ambiguous cases are where there is an exact match on address but a mismatch or partial match on DOB or name.
# MAGIC 
# MAGIC This makes sense - our model doesn't know the difference between a nickname/alias or two family members at the same address. 
# MAGIC 
# MAGIC We can explore this further by going back to the comparison viewer dashboard above and use the dropdowns to select comparisons where name does not match - do any of these end up with a high match weight? (_No, they don't_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Clusters
# MAGIC > **Once we're happy with our model, we probably have a rough idea of what threshold match score to use to define a match (e.g. >0.9).**
# MAGIC > 
# MAGIC >**Spot checking a sample of clusters with `cluster_studio_dashboard` we can check for a couple of things, the biggest risk being due to false positives (linking records that aren't genuine matches):**
# MAGIC > - **Are there any "hairball" clusters that have become overly large because several genuine clusters have been merged?**
# MAGIC > - **Ideal clusters will be fully-connected (every node is directly linked to every other) - are there clusters that appear to be two or more subclusters only connected by a single (suspicious) edge?**

# COMMAND ----------

clusters = linker.cluster_pairwise_predictions_at_threshold(
    predictions, 
    threshold_match_probability=0.8
)

# COMMAND ----------

path = f"/dbfs/Users/{username}/cluster_studio.html"
linker.cluster_studio_dashboard(
    predictions, 
    clusters, 
    out_path=path, 
    sampling_method="by_cluster_size", 
    overwrite=True
)

with open("/dbfs" + path, "r") as f:
    html=f.read()    
displayHTML(html)

# COMMAND ----------


