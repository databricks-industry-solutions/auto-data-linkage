# Databricks notebook source
# MAGIC %md
# MAGIC # Splink on Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is entity resolution?
# MAGIC 
# MAGIC Entity resolution is the process of working out whether multiple records in a dataset are referencing the same real-world thing. This could be anything - a person, an organization, or any kind of physical object. Traditional approaches to solving this problem have involved writing business rules into data processing logic, but this can quickly become unmanageable with thousands of rules codified to address specific edge cases. This results in a process that is brittle, opaque, and complex. 
# MAGIC 
# MAGIC Modern approaches to entity resolution revolve around machine learning and are capable of easily adapting to new data and scaling to huge volumes - however modern approaches require modern infrastructure, with traditional data warehouses unable to handle this work.
# MAGIC 
# MAGIC Splink is an open source entity resolution library developed by the United Kingdom's Ministry of Justice, in use in the public and private sector across the world. Splink uses an unsupervised learning algorithm in conjunction with a set of heuristics to simplify the record linking process. Splink can be used in conjunction with the Databricks Lakehouse architecture to provide a simple, secure, and scalable solution to entity resolution.
# MAGIC 
# MAGIC ***
# MAGIC 
# MAGIC In this Notebook we will walk through a simple example of setting up Splink on Databricks and applying it to a dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up Splink on Databricks
# MAGIC 
# MAGIC - Installing Splink
# MAGIC - Library imports
# MAGIC - Enabling Splink on Databricks
# MAGIC - Create working catalog/schema

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing Splink
# MAGIC 
# MAGIC We'll start by `pip` installing Splink from a repository:

# COMMAND ----------

# MAGIC %pip install splink --quiet

# COMMAND ----------

# DBTITLE 1,Get current username
username = spark.sql('select current_user() as user').collect()[0]['user']
db_name = username.replace(".", "_").replace("@", "_")
db_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing Libraries
# MAGIC 
# MAGIC We can now import the Splink library and the relevant sub-modules:
# MAGIC * The `SparkLinker` class for modelling work
# MAGIC * The `enable_splink` built-in method which configures Splink on Databricks all in one go
# MAGIC * The `spark_comparison_library` for built-in string comparison methods
# MAGIC * The `splink_mlflow` sub-module for built-in wrappers for MLFlow

# COMMAND ----------

from splink.spark.spark_linker import SparkLinker
from splink.databricks.enable_splink import enable_splink
import splink.spark.spark_comparison_library as cl
from utils.mlflow import splink_mlflow

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolving duplicates in a dataset of people
# MAGIC 
# MAGIC We'll be working with a dataset containing a list of individuals enriched with demographic information. We suspect that there are non-identical duplicates in this list, so we'll use Splink to set up deterministic and probabilistic rules to identify these.
# MAGIC 
# MAGIC We have over 162,000 data points, with each of them having the following information available about the individuals (although null values do exist in the data!):
# MAGIC * `name` - the full name of the individual
# MAGIC * `country_of_residence` the country in which they live in at the time of entry
# MAGIC * The `nationality` of the individual at the time of entry
# MAGIC * The `date_of_birth` of the individual
# MAGIC * The `forenames` of the individual
# MAGIC * The `surname` of the individual
# MAGIC * Individual address information about their residence:
# MAGIC   * `postal_code`
# MAGIC   * `premises`
# MAGIC   * `address_line_1`
# MAGIC   * `locality`
# MAGIC   * `region`
# MAGIC * A unique identifier `uid`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading and inspecting the data
# MAGIC 
# MAGIC We use spark to load the Delta table where the data resides.

# COMMAND ----------

# MAGIC %run
# MAGIC ../setup/data-setup

# COMMAND ----------

table_name = "companies_with_officers"
table_path = f"/mnt/source/splinkdata/delta/{table_name}"
data = spark.read.load(table_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's confirm the number of rows is as expected

# COMMAND ----------

data.count()

# COMMAND ----------

# MAGIC %md
# MAGIC And let's inspect the first 1000 rows to get a feel for the data.

# COMMAND ----------

data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC As an example, take a look at these two records - it would be reasonable to suggest that these are the same people. But how do we indentify this without trawling through the thousands of entries we have?

# COMMAND ----------

data.filter("uid = 23636 OR uid = 122382").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory data analysis with Splink
# MAGIC 
# MAGIC Splink provides interactive charts out of the box to explore our data (and others, which we'll look at later). Let's check the number of missing values per column and profile some columns to demonstrate.

# COMMAND ----------

# MAGIC %md
# MAGIC First, we'll instantiate a `linker` object with our dataset. Note we need to give the `SparkLinker` object a `spark` instance which is automatically defined in a Databricks notebook.

# COMMAND ----------

linker = SparkLinker(data, spark=spark)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now plot a chart to show the number of missing values per column in a single line:

# COMMAND ----------

linker.missingness_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC And we'll profile the `nationality` and `postal_code` columns:

# COMMAND ----------

data.columns

# COMMAND ----------

linker.profile_columns("nationality")

# COMMAND ----------

linker.profile_columns("postal_code")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Working with Splink

# COMMAND ----------

# MAGIC %md
# MAGIC ### How does Splink work
# MAGIC 
# MAGIC [Splink](https://moj-analytical-services.github.io/splink/index.html) is a Python package for probabilistic record linkage (entity resolution).
# MAGIC 
# MAGIC The core linkage algorithm is an implementation of Fellegi-Sunter's model of record linkage, with various customisations to improve accuracy.
# MAGIC 
# MAGIC There is a very succinct [blog post](https://www.robinlinacre.com/maths_of_fellegi_sunter/) summarising the Fellegi-Sunter model, and further context is provided in [this related paper](https://imai.fas.harvard.edu/research/files/linkage.pdf).
# MAGIC 
# MAGIC Suffice to say for this exercise, we'll be looking to estimate the probability of two records matching, given a set of criteria and constraints, to fulfil the following formula based on the Bayes Theorem:
# MAGIC 
# MAGIC \\(Pr(records\ match|some\ attributes\ match) = \frac{Pr(some\ attributes\ match|records\ match)Pr(records\ match)}{Pr(some\ attributes\ match)}\\)
# MAGIC 
# MAGIC which is the same as
# MAGIC 
# MAGIC \\(Pr(records\ match|some\ attributes\ match) = \frac{Pr(some\ attributes\ match|records\ match)Pr(records\ match)}{Pr(some\ attributes\ match|records\ match)Pr(records\ match)+Pr(some\ attributes\ match|records\ don't\ match)Pr(records\ don't\ match)}\\)
# MAGIC 
# MAGIC To estimate this probability, we'll define a few variables:
# MAGIC 
# MAGIC * \\(\lambda = Pr(records\ match)\\), i.e. our _prior_ belief of any two records matching
# MAGIC * A probability \\(m\\) with two levels:
# MAGIC   * \\(m_{0}=Pr(some\ attributes\ don't\ match|records\ match)\\)
# MAGIC   * \\(m_{1}=Pr(some\ attributes\ match|records\ match)\\)
# MAGIC * A probability \\(u\\) with two levels:
# MAGIC   * \\(u_{0}=Pr(some\ attributes\ don't\ match|records\ don't\ match)\\)
# MAGIC   * \\(u_{1}=Pr(some\ attributes\ match|records\ don't\ match)\\)
# MAGIC * \\(\gamma\\), a measure of similarity between values in a column
# MAGIC 
# MAGIC This turns our equation above to a much simpler form:
# MAGIC 
# MAGIC \\(Pr(records\ match|some\ attributes\ match)=\frac{m_{1}\lambda}{m_{1}\lambda+u_{1}(1-\lambda)}\\)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting up Splink for predictions
# MAGIC 
# MAGIC 
# MAGIC Splink estimates the \\(m\\) and \\(u\\) probabilities for us using the [expectation-maximilisation algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm), but we need to provide a few things ourselves.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Blocking rules for predictions
# MAGIC 
# MAGIC If we were to compare every record in our dataset against every other record, we would be looking at making over 26 _billion_ estimates. Splink allows you to greatly reduce this comparison space by implementing _blocking rules_ that basically constrain how two records must relate to each other in order to even be considered a potential match. A blocking rule is formulated as a SQL expression, and can be arbitrarily complex. For example, you could create record comparisons where the initial of the first name and the surname match. [This page](https://moj-analytical-services.github.io/splink/demos/03_Blocking.html) goes into more detail about blocking rules and their use.
# MAGIC 
# MAGIC To demonstrate this, we'll set up three rules:
# MAGIC 1. Full name and date of birth are an exact match
# MAGIC 2. Nationality, locality, premises and region are an exact match
# MAGIC 3. Address, post code and surname are an exact match
# MAGIC 
# MAGIC By doing this, we've reduced the comparison space from 26 billion pairs to some 146,000 - a significant improvement!

# COMMAND ----------

blocking_rules_to_generate_predictions = [
  "l.name = r.name and l.date_of_birth = r.date_of_birth",
  "l.nationality = r.nationality and l.locality = r.locality and l.premises = r.premises and l.region = r.region",
  "l.address_line_1 = r.address_line_1 and l.postal_code = r.postal_code and l.surname = r.surname",
]


# COMMAND ----------

# MAGIC %md
# MAGIC We can visualise these blocking rules using a built-in method within Splink by adding the rules as a setting to our `linker` object. Additionally, we'll define a few more settings that will help us later:
# MAGIC * `retain_intermediate_calculation_columns` and `retain_matching_columns` are to allow us to visualise the predictions at the end
# MAGIC * `link_type` is set to only deduplicate (as opposed to link multiple datasets) as we only have one dataset we're trying to deduplicate
# MAGIC * `unique_id_column_name` tells Splink the unique identifier column of our dataset
# MAGIC * `em_convergence` is setting a convergence threshold expectation-maximalisation algorithm

# COMMAND ----------


settings = {
  "retain_intermediate_calculation_columns": True,
  "retain_matching_columns": True,
  "link_type": "dedupe_only",
  "unique_id_column_name": "uid",
  "em_convergence": 0.01,
  "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions
}

linker.initialise_settings(settings)

# COMMAND ----------

linker.cumulative_num_comparisons_from_blocking_rules_chart(blocking_rules_to_generate_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Here we confirm the 140,000 number quoted above, with our second rule contributing the most to the total number of comparisons.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Comparison methods for each column
# MAGIC 
# MAGIC Splink provides out-of-the-box algorithms for comparing strings with its [`spark_comparison_library`](https://moj-analytical-services.github.io/splink/comparison_library.html), which may be familiar already:
# MAGIC 
# MAGIC 
# MAGIC ##### 1. Exact match
# MAGIC As the names suggests, `exact_match` is a method that returns a binary value depending on whether two values are exactly the same or not. This might be useful if we don't want to leave any wiggle room for misspellings, for example.
# MAGIC 
# MAGIC 
# MAGIC ##### 2. Levenshtein distance
# MAGIC The `levenshtein_at_thresholds` method gives a measure of how similar or not two strings are, and it considers the number of "edits" needed to make to get from one string to another. Edits are defined as inserts, deletes or changes of a single character. This method might be useful for columns where we expect differences to be due to mis-typing a letter on a keyboard.
# MAGIC 
# MAGIC 
# MAGIC ##### 3. Jaccard similarity
# MAGIC The `jaccard_at_thresholds` method gives the Jaccard index of two strings, which is defined as the number of letters they share divided by their combined letters; mathematically, the intersection of the two sets divided by their union. This method might be useful if we suspect there might be differences in how localities are defined in different datasets - think _Greater London_ vs _The Greater London Area_, for instance.
# MAGIC 
# MAGIC 
# MAGIC ##### 4. Jaro-Winkler similarity
# MAGIC The `jaro_winkler_at_thresholds` method gives the Jaro-Winkler similarity score between two strings, considering the relative position of each character in the string - characters are considered a match if they are the same character and within 2 characters of the same location in the comparison string. 
# MAGIC 
# MAGIC 
# MAGIC ##### A note on thresholds
# MAGIC 
# MAGIC Notice that similarity measure _2-4_ above all have an `*_at_thresholds` suffix. This is because Splink allows us to optionally define numeric thresholds for each of these, and creates comparison levels within each column. For example, given a Levenshtein rule with threshold of 2 on `firstname`, Splink will create three cardinal values for the similarity:
# MAGIC 1. If they're an exact match (great match)
# MAGIC 2. If they're not an exact match, but have a Levenshtein distance of at most 2 (good match)
# MAGIC 3. All others (bad match)
# MAGIC 
# MAGIC _Note: This effectively defines our \\(\gamma\\) value from above._

# COMMAND ----------

# MAGIC %md
# MAGIC We'll create a list of comparison methods for some columns of interest for demonstration and set arbitrary Levenshtein distances for all of our columns of interest.

# COMMAND ----------

comparisons = [
  cl.levenshtein_at_thresholds("surname", 10),
  cl.levenshtein_at_thresholds("forenames", 10),
  cl.levenshtein_at_thresholds("address_line_1", 5),
  cl.levenshtein_at_thresholds("country_of_residence", 1),
  cl.levenshtein_at_thresholds("nationality", 2)
]

# COMMAND ----------

# MAGIC %md
# MAGIC We'll now add these comparisons to our `linker` object's settings:

# COMMAND ----------

settings.update({"comparisons": comparisons})

linker.initialise_settings(settings)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. An estimate of the probability that any two given pairs are a match
# MAGIC 
# MAGIC Recall our formula from above, and in particular the \\(\lambda\\) parameter that was effectively the _prior_ to our Bayes formula, i.e. \\(Pr(records\ match)\\). We need some way of giving Splink an initial probability that two pairs are a match. We can do this by defining a set of deterministic rules that will help Splink do this.
# MAGIC 
# MAGIC We'll set up three rules to demonstrate:
# MAGIC * Names are a match and the Levenshtein distance of their dates of birth is less than or equal to 1
# MAGIC * Addresses are an exact match and the Levenshtein distances of their names are less than or equal to 5
# MAGIC * Their names are an exact match and the Levenshtein distances of their addresses are less than or equal to 5

# COMMAND ----------

deterministic_rules = [
  "l.name = r.name and levenshtein(r.date_of_birth, l.date_of_birth) <= 1",
  "l.address_line_1 = r.address_line_1 and levenshtein(l.name, r.name) <= 5",
  "l.name = r.name and levenshtein(l.address_line_1, r.address_line_1) <= 5",
]

# COMMAND ----------

# MAGIC %md
# MAGIC Additionally, we need to provide a sense (by which I mean, a number estimate) of how _well_ these rules do at defining true matches, according to our prior belief, for which Splink let's us give a _recall_ value to these rules. Let's demonstrate and estimate our \\(\lambda\\) given the above rules and a recall value of 80%:

# COMMAND ----------

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.8)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estimating model parameters
# MAGIC 
# MAGIC We now have everything to estimate our other parameters \\(u\\) and \\(m\\) (from the equations above) to eventually be able to generate predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Estimating the \\(u\\) probability
# MAGIC 
# MAGIC Recall that the \\(u\\) probability is the probability of observed values (which we now know are comparison levels \\(\gamma\\)) given that the records do __not__ match. This can be estimated using Splink's `estimate_u_using_random_sampling` method:

# COMMAND ----------

linker.estimate_u_using_random_sampling(target_rows=1e7)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Estimating the \\(m\\) probability
# MAGIC 
# MAGIC Recall that the \\(u\\) probability is the probability of observed values (which we now know are comparison levels \\(\gamma\\)) given that the records __do__ match. Without ground-truth labels, this can be estimated using the expectation-maximalisation algorithm with Splink's built-in `estimate_parameters_using_expectation_maximisation` method. For this, we need to run at least two passes of estimations, where we "fix" one or more columns and estimate \\(m\\) for the rest. We'll then need to fix other columns to estimate \\(m\\) for the first columns we fixed. Let's demonstrate:

# COMMAND ----------

# MAGIC %md
# MAGIC We'll first fix `name` and `date_of_birth` to match in order to estimate the other columns' \\(m\\) probabilities.

# COMMAND ----------

training_rule = "l.name = r.name and l.date_of_birth = r.date_of_birth"

# COMMAND ----------

linker.estimate_parameters_using_expectation_maximisation(training_rule)

# COMMAND ----------

# MAGIC %md
# MAGIC We'll now fix `surname` and `forenames` and estimate \\(m\\) for the remaining columns:

# COMMAND ----------

training_rule = "l.surname = r.surname and l.forenames = r.forenames"

# COMMAND ----------

linker.estimate_parameters_using_expectation_maximisation(training_rule)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualising our estimated probabilities
# MAGIC 
# MAGIC We can use Splink's built-in `m_u_parameters_chart` method to visualise the interpretations of how the \\(m\\) and \\(u\\) probabilities were estimated:

# COMMAND ----------

linker.m_u_parameters_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making predictions using our model
# MAGIC 
# MAGIC We can now make predictions using our model to predict duplicates in our dataset. This will return a Pandas dataframe with pair-wise comparisons between individuals in our dataset. To reduce the number of rows we end up printing out, we can also pass a `threshold_match_probability` so we only display matches above that threshold.
# MAGIC 
# MAGIC Below we'll predict duplicates and display the results in a Spark dataframe to make use of its more interactive view in a Databricks notebook.

# COMMAND ----------

predictions = linker.predict()
pdf_predictions = predictions.as_pandas_dataframe()
spark.createDataFrame(pdf_predictions).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualising model results
# MAGIC 
# MAGIC Splink lets us visualise the rationale behind model predictions using a built-in waterfall chart. We need to pass a dictionary to Splink's `waterfall_chart` method, so we'll convert our predictions to a dict before calling it.

# COMMAND ----------

dict_predictions = predictions.as_record_dict(limit=20)
linker.waterfall_chart(dict_predictions)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving our results to an MLFlow experiment
# MAGIC 
# MAGIC Splink integrates with [MLFlow](https://mlflow.org/), which is an open source end-to-end ML lifecycle management tool that is also available in Databricks as a managed service. With one line, we can log our model, results and artifacts to an MLFlow experiment for later re-use and comparison with other runs.
# MAGIC 
# MAGIC MLFlow also lets us store the model in a format that it can be used simply by a third party. 

# COMMAND ----------

 # Log model and parameters
with mlflow.start_run() as run:
    splink_mlflow.log_splink_model_to_mlflow(linker, "linker")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC After the model is stored, the next step is to retrieve the model and use it - MLFlow uses standardised APIs - all that is necessary is to call `predict()` on the returned object.

# COMMAND ----------

run_id = run.info.run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/linker")

splink_results = loaded_model.predict(data)
splink_results.as_spark_dataframe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC MLFlow simplifies things by wrapping models in a layer of MLFlow - this provides a model agnostic approach, whereby one can completely replace the underlying model without impacting on downstream processes because those downstream processes are only calling `.predict()` on an MLFlow model. 
# MAGIC 
# MAGIC However, one can still access the underlying model, which means one can still access all of the built in Splink functions, with just a little bit of extra code. 
# MAGIC 
# MAGIC 
# MAGIC (NB - the underlying Splink model only becomes available after it is used to make a prediction)

# COMMAND ----------

dict_predictions = splink_results.as_record_dict(limit=20)
loaded_model.unwrap_python_model().linker.waterfall_chart(dict_predictions)
