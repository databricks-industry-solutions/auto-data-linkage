# Databricks notebook source
# MAGIC %md # Introducing Databricks ARC
# MAGIC
# MAGIC
# MAGIC Databricks ARC (Automated Record Connector) is a solution accelerator by Databricks that performs simple, automated, scalable probabilistic data de-duplication and linking. This example notebook shows how to use ARC to find near-duplicates in a dataset, explain some of the optional arguments which can fine tune performance, and visually explore the linking model ARC produces. 
# MAGIC
# MAGIC For detailed information on what is happening under the hood, please visit the documentation [**here**](https://databricks-industry-solutions.github.io/auto-data-linkage/). 
# MAGIC
# MAGIC A brief explanation of the process is below:
# MAGIC - ARC uses an unsupervised linking engine ([Splink](https://github.com/moj-analytical-services/splink)) to find near duplicates in the input data. 
# MAGIC - As an unsupervised process, the linking performance is dependent on the user provided options to configure the linking model.
# MAGIC - ARC uses a combination of heuristics and Bayesian hyperparameter optimisation via [Hyperopt](https://hyperopt.github.io/hyperopt/) to search through a large number of options to choose the best model. The evaluation metric for a "good" model is explained in the documentation. 
# MAGIC - This optimisation is an iterative process which improves the model over time - as a rule of thumb, the longer ARC runs for the better model it produces. 
# MAGIC - Each iteration is logged for reference and audit purposes in [MLFlow](https://mlflow.org/).
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Requirements
# MAGIC This notebook was tested on a cluster configured with 8 workers, 12.2 LTS ML Runtime. The VM type was General Purpose (exact type will depend on your cloud provider), 14GB Memory & 4 cores.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install ARC from PyPI

# COMMAND ----------

# MAGIC %pip install --quiet databricks-arc

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load ARC

# COMMAND ----------

from arc.autolinker import AutoLinker

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read test data
# MAGIC
# MAGIC This is synthetic test data generated using [Febrl](https://github.com/Rehket/FEBRL). There are 2 files. *main_records.csv* contains 100,000 records without duplicates. *duplicates.csv* contains 3,000 near duplicates to the other 100,000 records. 
# MAGIC
# MAGIC The column *rec2_id* provides the True record ID - records that share the same *rec2_id* value are duplicates of each other. *rec_id* is a row level unique ID.
# MAGIC
# MAGIC As we are loading from a CSV file, we also need to repatition the data to make use of Spark's parallelism.
# MAGIC
# MAGIC We will use ARC to link these two files together.

# COMMAND ----------

import os
#we need the file: prefix for spark to correctly read the data file from a Databricks Repo.
duplicates = spark.read.csv(f"file:{os.getcwd()}/data/duplicates.csv", header=True).repartition(200).cache() 
main_records = spark.read.csv(f"file:{os.getcwd()}/data/main_records.csv", header=True).repartition(200).cache()

print(f"There are {main_records.count} records in main_records.csv.")
print(f"There are {duplicates.count} records in duplicates.csv.")

display(duplicates)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform Auto-linking
# MAGIC
# MAGIC ARC can work with only a single argument - the data! For linking, ARC requires a list of 2 Spark DataFrames. In most cases, you aren't going to have a a column like *recid* which you can use to tell you which records are duplicates - if you did, you wouldn't need to deduplicate your data! 
# MAGIC
# MAGIC For the first example, we will drop the *uid* and *recid* columns. This will simulate a pair of datasets which don't have a primary key or a True record id.
# MAGIC
# MAGIC **NB** - ARC is an interative process, but for the sake of demonstration here we will only run a single iteration. This is controlled by the *max_evals* argument (this has a default of 5. In testing, we found good performance with 100 iterations). This should take about 5 minutes to run.
# MAGIC

# COMMAND ----------

linker1 = AutoLinker()

link_data = [
  duplicates.drop("rec2_id", "rec_id")
  ,main_records.drop("rec2_id", "rec_id")
  ]

linker1.auto_link(
  data=link_data
  ,max_evals=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ARC's underlying linking engine is incredibly verbose in it's output! This tells us in detail what stage the linking is at and what parameters are currently being evaluted. 
# MAGIC
# MAGIC You will note that just underneath the previous cell it mentions logging 1 run to an experiment in MLFlow. 
# MAGIC
# MAGIC Clicking on the *run* hyperlink will show you what has been automatically logged in MLflow for you. More details of what is logged is available in the documentation.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Let's objectively evaluate the model using the *recid* true values.
# MAGIC
# MAGIC For this, we will also use some additional argument to illustrate their use. 
# MAGIC - **unique_id** - this argument tells ARC the data contains a row level unique ID column and to ignore it (this won't be useful in deciding if 2 records are the same). If missing, ARC will create a column called *uid* for you.
# MAGIC - **attribute_columns** - this argument tells ARC which columns should be used to determine if records are the same or not. You can use this if the data contains information which is not helpful for matching. For example, if you have a datset where each record has the exact same value, this will not help in determining if 2 records are the same. In this case, we are ignoring state and suburb values
# MAGIC - **true_label** - this argument will turn on ARC's objective evaluation functionality. It will calculate classification accuracy metrics and store in MLFlow so you can objectively measure how well the model is performing (for example Precision, Recall & F1 score )

# COMMAND ----------

linker2 = AutoLinker()
link_data = [
  duplicates
  ,main_records
  ]

linker2.auto_link(
  data=link_data
  ,unique_id="rec_id"
  ,attribute_columns = [
    'given_name',
    'surname',
    'street_number',
    'address_1',
    'address_2',
    'postcode',
    'date_of_birth'
  ]
  ,true_label="rec2_id"
  ,max_evals=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Inspecting the MLFlow Run from the above cell will now show a additional set of metrics that have been logged

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Linking data with different schemas (experimental).
# MAGIC
# MAGIC Data linking is often between datasets of different schemas. This maybe a difference in naming convention (i.e. ```address_1``` and ```address_line_1``` or through completely new information. For example, a common usecase of data linking is to combine dataset which contain different information - for example, in two datasets about people, one may contain employment details whilst another contains education details. 
# MAGIC
# MAGIC For optimal performance, the input schemas for linking should be the same, but ARC is capable of linking data with divergent schemas. It does this by applying some hueristics to look at which columns have the most overlap in the two inputs.
# MAGIC
# MAGIC **NB** - when setting attribute columns or unique ID arguments when linking data with different schemas, the respective columns must be present in both input dataframes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get clusters from best model
# MAGIC
# MAGIC Once ARC has finished finding the best set of parameters for the model, we need to inspect the output. This is referred to as the  "clusters". In this output, ARC appends a new column to the dataset called *cluster_id* - records which share a cluster ID are the same, according to ARC (the quality of the results depends on how many iterations ARC has run for.)

# COMMAND ----------

clusters = linker2.best_clusters_at_threshold()
clusters.orderBy("cluster_id").display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ```best_clusters_at_threshold()``` has a single optional argument; **threshold** (default=0.8). 
# MAGIC
# MAGIC threshold takes a value between 0 and 1, and represents a confidence level for when ARC will consider 2 records the same; the lower the threshold, the lower the confidence ARC will need to assert they are the same. 
# MAGIC
# MAGIC
# MAGIC The threshold argument is used to control the *precision* of the clustering - setting this near to 1 will result in finding very similar records, with a low likelihood of mistakes - but also a low likelihood of finding less similar duplicates. Conversely, a lower number will give better coverage of finding less similar duplicates, but will increase the probability of mistakes. 
# MAGIC
# MAGIC A good rule of thumb is to stick between 0.7 and 1.0.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve pairwise predictions
# MAGIC
# MAGIC We can also inspect what pairs of records ARC has considered, and the probability it has assigned that each pair is a match. To understand how a candidate pair is presented to ARC for scoring, please see the documentation.
# MAGIC
# MAGIC This view can be helpful for evaluating ARCs performance or further inspecting its decision making to gain a deeper understanding.

# COMMAND ----------

predictions = linker1.best_predictions_df.as_spark_dataframe()
predictions.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's inspect how the ML model is making it's decisions.
# MAGIC
# MAGIC The ```match_weights_chart()``` shows how much importance ARC is putting on each comparison within each column. At a glance this lets you see which attributes are impacting your model most - dark green bars are comparisons ARC is using as strong signs of a match. Dark red bars are the opposite - strong signs of not a match. For more information on this, see the Splink [documentation](https://moj-analytical-services.github.io/splink/).

# COMMAND ----------

linker1.match_weights_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### We can also inspect individual record's matching decisions
# MAGIC
# MAGIC These waterfall charts give a granular breakdown of how ARC is making it's decisions. For more information on this, see the Splink [documentation](https://moj-analytical-services.github.io/splink/).

# COMMAND ----------

records_to_view  = linker1.best_predictions_df.as_record_dict(limit=5)
linker1.best_linker.waterfall_chart(records_to_view, filter_nulls=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Advanced - accessing the underlying Splink object from ARC.
# MAGIC
# MAGIC The above charts highlight two of Splinks visualisation features. Splink is a richly featured tool with a lot of power and customisability. It is possible to directly access this through the ```autolinker.best_linker``` attribute. 
# MAGIC
# MAGIC This enables a workflow whereby ARC is used to jumpstart a deduplication project - use ARC to build a baseline model through automation, and then manually iterate and improve, increasing your efficiency by using machines to do the initial, easier work.

# COMMAND ----------

linker = linker1.best_linker
linker.m_u_parameters_chart()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Clean up 
# MAGIC
# MAGIC ARC creates an MLFlow experiment each time you run it to track performance across the optimisation process. This can lead to a lot of MLFlow experiments being created in a short time. ARC has a utility method to clean up. 
# MAGIC
# MAGIC **BE WARNED** - this will delete *all* your default named MLFlow experiments. Make sure that you have renamed or otherwise saved any experiments you wish to keep!

# COMMAND ----------

from arc.utils import utils 

utils.delete_all_default_linking_experiments(spark)
