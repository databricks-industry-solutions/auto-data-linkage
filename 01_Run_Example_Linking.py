# Databricks notebook source
# MAGIC %md ## Project Description
# MAGIC
# MAGIC Databricks ARC (Automated Record Connector) is a solution accelerator by Databricks that performs highly scalable probabilistic data de-duplication without the requirement for any labelled data or subject matter expertise in entity resolution.
# MAGIC
# MAGIC ARC's linking engine is the UK Ministry of Justice's open-sourced entity resolution package, [Splink](https://github.com/moj-analytical-services/splink). It builds on the technology of Splink by removing the need to manually provide parameters to calibrate an unsupervised de-duplication task, which require both a deep understanding of entity resolution and good knowledge of the dataset itself. The way in which ARC achieves this is detailed in the table below:
# MAGIC
# MAGIC | **Parameter**           | **Splink**                                                                                                                                            | **ARC**                                                                                                                                                                                 |
# MAGIC |-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC | Prior match probability | User to provide SQL-like statements for “deterministic rules” and a recall score for Splink to estimate the prior probability that two records match. | Automatically set prior probability to $$\frac{1}{N}$$.                                                                                                                                                  |
# MAGIC | Training rules          | User to provide SQL-like statements for a series of rules which trains the m probability values for each column.                                      | Automatically generate training rule statements such that each column is trained.                                                                                                       |
# MAGIC | Comparisons             | User to provide distance functions and thresholds for each column to compare.                                                                         | Automatically optimise multi-level parameter space for functions and thresholds.                                                                                                        |
# MAGIC | Blocking rules          | User to provide SQL-like statements to determine the possible comparison space and reduce the number of pairs to compare.                             | User to provide a parameter for the maximum number of pairs they’re willing to compare; Arc identifies all possible blocking rules within that boundary and optimises for the best one. |
# MAGIC
# MAGIC
# MAGIC ### Parameter Optimisation
# MAGIC
# MAGIC Arc uses Hyperopt (http://hyperopt.github.io/hyperopt/) to perform a Bayesian search to find the optimal settings, where optimality is defined as minimising the entropy of the data after clustering and standardising cluster record values. The intuition here is that as we are linking different representations of the same entity together (e.g. Facebook == Fakebook), then standardising data values within a cluster will reduce the total number of data values in the dataset.
# MAGIC
# MAGIC To achieve this, Arc optimises for a custom information gain metric which it calculates based on the clusters of duplicates that Splink predicts. Intuitively, it is based on the reduction in entropy when the data is split into its clusters. The higher the reduction in entropy in the predicted clusters of duplicates predicted, the better the model is doing. Mathematically, we define the metric as follows:
# MAGIC
# MAGIC Let the number of clusters in the matched subset of the data be *c*.
# MAGIC
# MAGIC Let the maximum number of unique values in any column in the original dataset be *u*.
# MAGIC
# MAGIC Then the "scaled" entropy of column *k*, *N* unique values with probability *P* is
# MAGIC
# MAGIC $$E_{s,k} = -\Sigma_{i}^{N} P_{i} \log_{c}(P_{i})$$
# MAGIC
# MAGIC Then the "adjusted" entropy of column *k*, *N* unique values with probability *P* is
# MAGIC
# MAGIC $$E_{a,k} = -\Sigma_{i}^{N} P_{i} \log_{u}(P_{i})$$
# MAGIC
# MAGIC The scaled information gain is
# MAGIC
# MAGIC $$I_{s} = \Sigma_{k}^{K} E_{s,k} - E'_{s,k}$$
# MAGIC
# MAGIC and the adjusted information gain is
# MAGIC
# MAGIC $$I_{a} = \Sigma_{k}^{K} E_{a,k} - E'_{a,k}$$
# MAGIC
# MAGIC where *E* is the mean entropy of the individual clusters predicted.
# MAGIC
# MAGIC The metric to optimise for is:
# MAGIC
# MAGIC $$I_{s}^{I_{a}}$$

# COMMAND ----------

# MAGIC %md
# MAGIC # Install ARC and Splink

# COMMAND ----------

# MAGIC %pip install --quiet databricks-arc splink sentence-transformers

# COMMAND ----------

# MAGIC %md
# MAGIC # Load ARC

# COMMAND ----------

from arc.autolinker import AutoLinker
import arc

# COMMAND ----------

# MAGIC %md
# MAGIC # Enable ARC

# COMMAND ----------

arc.enable_arc()

# COMMAND ----------

# MAGIC %md
# MAGIC # Read test data

# COMMAND ----------

import os
data = spark.read.csv(f"file:{os.getcwd()}/data/arc_febrl1.csv", header=True) #we need the file: prefix for spark to correctly read the data file

data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Perform Auto-linking

# COMMAND ----------

autolinker = AutoLinker()

attribute_columns = ["given_name", "surname", "street_number", "address_1", "address_2", "suburb", "postcode", "state", "date_of_birth"]

autolinker.auto_link(
  data=data,                                                         # Spark DataFrame of data to deduplicate
  attribute_columns=attribute_columns,                               # List of column names containing attribute to compare
  unique_id="uid",                                                   # Name of the unique identifier column
  comparison_size_limit=100000,                                      # Maximum number of pairs to compare
  max_evals=10                                                       # Number of trials to run during optimisation process
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Get clusters from best model

# COMMAND ----------

clusters = autolinker.best_clusters_at_threshold(threshold=0.3)
clusters.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieve pairwise predictions

# COMMAND ----------

autolinker.best_predictions_df.as_spark_dataframe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Splink functionality

# COMMAND ----------

autolinker.cluster_viewer()

# COMMAND ----------

autolinker.match_weights_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC # Retrieve the best model directly from `autolinker`

# COMMAND ----------

linker = autolinker.best_linker

# COMMAND ----------

linker.profile_columns("surname")
