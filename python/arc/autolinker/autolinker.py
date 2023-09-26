from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark
from ..sql import functions as arcf
import os

if "DATABRICKS_RUNTIME_VERSION" in os.environ:
  from dbruntime.display import displayHTML

  

import splink
from splink.spark.spark_linker import SparkLinker
import splink.spark.spark_comparison_library as cl

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from . import splink_mlflow

import numpy as np
import math
import random
from datetime import datetime
import itertools


from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             completeness_score, fowlkes_mallows_score, homogeneity_score, 
                             mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score)

import typing

import mlflow
from mlflow.tracking import MlflowClient


class AutoLinker:
  """
  Class to create object arc for automated data linking.
  
  :param spark: Spark instance used
  :param catalog: Unity Catalog name, if used, for intermediate tables
  :param schema: Schema/database name for intermediate tables
  :param experiment_name: Location or simply name of the MLflow experiment for storing trial run results
  :param training_columns: Valid list of columns (or column names separated by '&' for AND combinations) of columns to be fixed during EM training
  :param deterministic_columns: Valid list of columns (or column names separated by '&' for AND combinations) of columns to be used for prior estimation

  Basic usage:
  
  .. code-block:: python

  >>> arc = AutoLinker(
  ...   catalog="splink_catalog",                # catalog name
  ...   schema="splink_schema",                  # schema to write results to
  ...   experiment_name="autosplink_experiment"  # MLflow experiment location
  ... )
  >>> arc.auto_link(
  ...   data=data,                               # dataset to dedupe
  ...   attribute_columns=["A", "B", "C", "D"],  # columns that contain attributes to compare
  ...   unique_id="id",                          # column name of the unique ID
  ...   comparison_size_limit=500000,            # Maximum number of pairs when blocking applied
  ...   max_evals=100                            # Maximum number of hyperopt trials to run
  ... )
  >>> arc.best_linker.m_u_parameters_chart()     # use Splink functionality out-of-the-box

  """
  
  
  def __init__(
    self,
    spark:str=None,
    catalog:str=None,
    schema:str=None,
    experiment_name:str=None,
    training_columns:list=None,
    deterministic_columns:list=None
  ) -> None:
    """
    Initialises an AutoLinker instance to perform automated record linking.
    """
    self.spark = spark 
    self.catalog = catalog
    self.schema = schema

    self.data = None
    
    self.experiment_name = experiment_name
    self.training_columns = training_columns
    self.deterministic_columns = deterministic_columns
    self.best_params = None
    self.clusters=None
    self.cluster_threshold=None
    self.original_entropies = dict()


    self._autolink_data = None
    self._cleaning_mode = None
    self.attribute_columns = None
    self.unique_id = None
    self.linker_mode = None


  def __str__(self):
    return f"AutoLinker instance working in {self.catalog}.{self.schema} and MLflow experiment {self.experiment_name}"

  
  def _calculate_dataset_entropy(
      self,
      data:pyspark.sql.DataFrame,
      attribute_columns:list,
      by_cluster:bool=False,
      base:int=0
      ) -> dict:
    """
    Method to calculate the average entropy of each attribute column in a dataset. Uses the custom arc_entropy_agg
    function from ARC's SQL built-ins to efficiently calculate entropy across all columns.

    Returns a dictionary with keys as columns and values as entropy in the columns.

    :param data: input dataframe with row per record
    :param column: (valid) name of column to calculate entropy on
    :param by_cluster: if True, it will calculate the average entropy when the dataset is split by clusters.
    :param base: base of the log in the entropy calculation
    """
    if by_cluster:
      cluster_groupby = "cluster_id"
    else:
      cluster_groupby = F.lit(1)


    entropies = data.fillna("null_").groupBy(cluster_groupby).agg(
        arcf.arc_entropy_agg(base, *attribute_columns).alias("ent_map")
      ).select(
        *(F.expr(f"mean(ent_map.{c}) as {c}") for c in attribute_columns)
      ).collect()[0]

    entropy_dict = {c: entropies[c] if entropies[c] else 0.0 for c in attribute_columns}
    
    return entropy_dict
  

  def _calculate_unsupervised_metrics(
      self,
      clusters:pyspark.sql.DataFrame,
      attribute_columns:list,
      base:int
      ) -> dict:
    """
    Method to calculate the chosen metric for unsupervised optimisation, the power ratio of information gains when
    calculated using a base of the number of clusters and a base of the maximum number of unique values in any 
    columns in the original data. Information gain is defined as the difference in average entropy of clustered
    records when split and the entropy of the data points that are being matched.

    Let the number of clusters in the matched subset of the data be :math:`c`

    Let the maximum number of unique values in any column in the original dataset be :math:`u`

    Then the "scaled" entropy of column :math:`k`, :math:`N` unique values with probability :math:`P` is

    :math:`E_{s,k} = -\Sigma_{i}^{N} P_{i} \log_{c}(P_{i})`

    Then the "adjusted" entropy of column :math:`k`, :math:`N` unique values with probability :math:`P` is

    :math:`E_{a,k} = -\Sigma_{i}^{N} P_{i} \log_{u}(P_{i})`

    The scaled information gain is

    :math:`I_{s} = \Sigma_{k}^{K} E_{s,k} - E'_{s,k}`

    and the adjusted information gain is

    :math:`I_{a} = \Sigma_{k}^{K} E_{a,k} - E'_{a,k}`

    where :math:`E'` is the mean entropy of the individual clusters predicted.

    The metric to optimise for is:

    :math:`I_{s}^{I_{a}}`

    Returns information gain metric, as dictionary.

    :param clusters: dataframe returned from linker clustering
    :param attribute_columns: list of valid column names containing attributes in the clusters dataset
    :param base: the base for the log when calculating the adjusted entropy
    """

    # Calculate count of rows in each cluster and rejoin
    cluster_counts = clusters.groupBy("cluster_id").count().withColumnRenamed("count", "_cluster_count")
    data = clusters.join(cluster_counts, on=["cluster_id"], how="left")
    
    # Number of non-singleton clusters (for entropy base)
    num_clusters = cluster_counts.filter("_cluster_count>1").count()

    # Filter on rows which have a match
    df_matched = data.filter("_cluster_count>1")

    # Calculate matched whole data entropy
    entropy_matched_scaled = self._calculate_dataset_entropy(df_matched, attribute_columns, by_cluster=False, base=num_clusters)
    entropy_matched_adjusted = self._calculate_dataset_entropy(df_matched, attribute_columns, by_cluster=False, base=base)
    
    # Calculate mean entropy by clusters
    entropy_cluster_mean_scaled = self._calculate_dataset_entropy(df_matched, attribute_columns, by_cluster=True, base=num_clusters)
    entropy_cluster_mean_adjusted = self._calculate_dataset_entropy(df_matched, attribute_columns, by_cluster=True, base=base)
 
    # Information Gain
    information_gain2_scaled = np.sum([entropy_matched_scaled[c] - entropy_cluster_mean_scaled[c] for c in attribute_columns])
    information_gain2_adjusted = np.sum([entropy_matched_adjusted[c] - entropy_cluster_mean_adjusted[c] for c in attribute_columns])

    # Power Ratio of scaled and adjusted information gains
    information_gain_power_ratio = math.pow(information_gain2_scaled, information_gain2_adjusted)
    
    unsupervised_metrics = {
      "information_gain_power_ratio": information_gain_power_ratio
    }

    return unsupervised_metrics


    
  def _generate_rules(
    self,
    columns:list
  ) -> list:
    """
    Method to create a list of Splink-compatible SQL statements from a list of candidate columns for rules.
    Can be used for simple blocking rules (no AND statement), deterministic rules or for training rules.
    
    Returns list of strings of Splink-compatible SQL statements.

    :param columns: valid attribute columns in the data set

    """

    rules = []
    for column in columns:
      if "&" in column:
        subcols = column.split("&")
        subcols = [subcol.strip() for subcol in subcols]

        rule = " AND ".join([f"l.{subcol} = r.{subcol}" for subcol in subcols])
      else:
        rule = f"l.{column} = r.{column}"

      rules.append(rule)

    return rules
  
  
  
  def _generate_candidate_blocking_rules(
    self,
    data:pyspark.sql.DataFrame,
    attribute_columns:list,
    comparison_size_limit:int,
    max_columns_per_and_rule:int=2,
    max_rules_per_or_rule:int=3,
  ) -> list:
    """
    Method to automatically generate a list of lists of blocking rules to test, given a user-defined limit for
    pair-wise comparison size. Uses appriximated method built in to ARC's SQL functions.

    Returns a nested list of lists with Splink-compatible blocking rule queries.

    :param data: input data with record-per-row
    :param attribute_columns: valid column names of data, containing all possible columns to block on
    :param comparison_size_limit: the maximum number of pairs we want to compare, to limit hardware issues
    :param max_columns_per_and_rule: the maximum number of column comparisons in a single rule to try
    :param max_rules_per_or_rule: the maximum number of rules comparisons in a composite rule to try

    """

    # replace null values with dummy
    data = data.fillna("null_")

    # Create blocking rules and size esimates
    df_rules = arcf.arc_generate_blocking_rules(data, max_columns_per_and_rule, max_rules_per_or_rule, *attribute_columns)

    # Filter on max
    rules = df_rules.filter(f"rule_squared_count < {comparison_size_limit}").collect()

    accepted_rules = [rule.splink_rule for rule in rules]

    # set deterministic rules to be 500th largest (or largest) blocking rule
    self.deterministic_columns = df_rules.orderBy(F.col("rule_squared_count")).limit(500).orderBy(F.col("rule_squared_count").desc()).limit(1).collect()[0]["splink_rule"]

          
    return accepted_rules

  
  
  def _create_hyperopt_space(
    self,
    data:pyspark.sql.DataFrame,
    attribute_columns:list,
    comparison_size_limit:int,
    max_columns_per_and_rule:int=2,
    max_rules_per_or_rule:int=3
  ) -> dict:
    """
    Method to create hyperopt space for comparison and blocking rule hyperparameters from list of columns.
    Takes a given (or generated) list of columns and produces a dictionary that can be converted to a comparison list later on,
    with function names and threhsolds, and a list (of lists) of blocking rules to test.

    Returns a dictionary for hyperopt parameter search

    :param data: input data with records per row
    :param attribute_columns: valid column names of data, containing all possible columns to compare
    :param comparison_size_limit: maximum number of pairs we want to compare, to limit hardware issues
    :param max_columns_per_and_rule: the maximum number of column comparisons in a single rule to try
    :param max_rules_per_or_rule: the maximum number of rules comparisons in a composite rule to try
    
    """

    # Generate candidate blocking rules
    self.blocking_rules = self._generate_candidate_blocking_rules(
      data=data,
      attribute_columns=attribute_columns,
      comparison_size_limit=comparison_size_limit,
      max_columns_per_and_rule=max_columns_per_and_rule,
      max_rules_per_or_rule=max_rules_per_or_rule
    )
    # Create comparison dictionary using Hyperopt sampling
    space = dict()
    comparison_space = dict()

    # Create hyperopt space for comparisons
    # TODO: make the threshold boundaries functions of the columns in some way
    for column in attribute_columns:
      comparison_space.update(
        {
          column: {
            "distance_function": hp.choice(
              f"{column}|distance_function", [
                {"distance_function": "levenshtein", "threshold": hp.quniform(f"{column}|levenshtein|threshold", 1, 5, q=1)},
                {"distance_function": "jaccard", "threshold": hp.uniform(f"{column}|jaccard|threshold", 0.7, 0.99)},
                {"distance_function": "jaro_winkler", "threshold": hp.uniform(f"{column}|jaro_winkler|threshold", 0.7, 0.99)}
              ]
            )
          }
        }
      )

    # Create hyperopt space for blocking rules
    space.update({
      "blocking_rules": hp.choice("blocking_rules", self.blocking_rules),
      "comparisons": comparison_space
    })
    return space
  
  
  
  def _drop_intermediate_tables(self):
    """
    Method to drop intermediate tables for clean consecutive runs.
    """
    # Drop intermediate tables in schema for consecutive runs
    tables_in_schema = self.spark.sql(f"show tables from {self.catalog}.{self.schema} like '*__splink__*'").collect()
    for table in tables_in_schema:
      try:
        self.spark.sql(f"drop table {self.catalog}.{self.schema}.{table.tableName}") 
      except:
        self.spark.sql(f"drop table {table.tableName}")
        
  
  def _clean_columns(
    self,
    data:pyspark.sql.DataFrame,
    attribute_columns:list):
    """
    Method to clean string columns (turn them to lower case and remove non-alphanumeric characters)
    in order to help with better (and quicker) string-distance calculations. If cleaning is 'all'
    (as is by default), it will automatically clean as much as it can.  If cleaning is 'none', it will do nothing.
    cleaning can also be a dictionary with keys as column names and values as lists of method strings.
    The currently available cleaning methods are turning to lower case and removing non-alphanumeric characters.

    Returns a Spark DataFrame.

    :param data: DataFrame containing the data to be cleaned
    :param attribute_columns: all attribute columns
    :param cleaning: string or dicitonary with keys as column names and values as list of valid cleaning method names.

    """

    cleaning = self._cleaning_mode
    # if cleaning is set to "none", don't do anything to the data
    if cleaning=="none":
      return data
    
    # if it's set to "all", turn it into a dictionary covering all possible cases
    if cleaning=="all":
      cleaning = {col: ["lower", "alphanumeric_only"] for col in attribute_columns}
      
    for col, methods in cleaning.items():
      # if column is not a string, skip it
      if not data.schema[col].dataType == StringType():
        continue
        
      for method in methods:
        if method=="alphanumeric_only":
          # replace column and only keep alphanumeric and space characters
          data = data.withColumn(col, F.regexp_replace(F.col(col), r"[^A-Za-z0-9 ]+", ""))
        
        elif method=="lower":
          data = data.withColumn(col, F.lower(F.col(col)))
          
    return data

  def _do_column_cleaning(
          self
          ,autolink_data
          ,linker_mode
          ,attribute_columns
  ) -> typing.Union[pyspark.sql.dataframe.DataFrame, list]:
    # Clean the data
    if linker_mode == "dedupe_only":
      output_data = self._clean_columns(autolink_data, attribute_columns)
    else:
      output_data = [
        self._clean_columns(autolink_data[0], attribute_columns),
        self._clean_columns(autolink_data[1], attribute_columns)
      ]
    return output_data

  def _convert_hyperopt_to_splink(
          self
  ) -> dict:
    """
    Method to convert hyperopt trials to a dictionary that can be used to
    train a linker model. Used for training the best linker model at the end of an experiment.
    Sets class attributes for best metric and parameters as well.
    Returns a dictionary.
    """
    # Set best params, metrics and results
    t = self.trials.best_trial
    self.best_metric = t['result']['loss']
    params = t['misc']['vals']
    blocking_rules = t['misc']['vals']['blocking_rules'][0]
    comparison_params = {}

    for k, v in params.items():
      if ("threshold" in k) & (len(v)>0):
        column, function, _ = k.split('|')
        comparison_params.update({
          column: {
            "function": function,
            "threshold": v[0]
          }
        })

    self.best_params = {
      "blocking_rules": self.blocking_rules[blocking_rules],
      "comparisons": comparison_params
    }

    best_comparisons = {}
    for k, v in comparison_params.items():
      best_comparisons.update({
        k: {"distance_function": {"distance_function": v['function'], "threshold": v['threshold']}}
      })
      
    best_params_for_rt = {
      "blocking_rules": self.best_params['blocking_rules'],
      "comparisons": best_comparisons
    }

    return best_params_for_rt


  def _create_comparison_list(
    self,
    space:dict
  ) -> list:
    """
    Method to convert comparisons dictionary generated by hyperopt to a Splink-compatible list of
    spark_comparison_library functions with the generated columns and thresholds.

    Returns list of spark_comparison_library method instances.

    :param space: nested dicitonary generated by create_comparison_dict

    """

    # Initialise empty list to populate
    comparison_list = []

    # Get comparison dict
    comparison_dict = space["comparisons"]

    # Loop through comparison dictionary and generate Splink-compatible comparison list
    for column in comparison_dict.keys():
      distance_function = comparison_dict[column]["distance_function"]["distance_function"]
      threshold = comparison_dict[column]["distance_function"]["threshold"]

      if distance_function=="levenshtein":
        comparison_list.append(cl.levenshtein_at_thresholds(column, threshold))
      elif distance_function=="jaccard":
        comparison_list.append(cl.jaccard_at_thresholds(column, threshold))
      elif distance_function=="jaro_winkler":
        comparison_list.append(cl.jaro_winkler_at_thresholds(column, threshold))
      else:
        raise ValueError(f"Unknown distance function {distance_function} passed.")

    return comparison_list
  
  
  def _randomise_columns(
    self,
    attribute_columns:list
  ) -> list:
    """
    Method to randomly select (combinations of) columns from attribute columns for EM training.
    Will try to pick 2 combinations of 2 (i.e AB and BC from ABC), but will default to 2 if there are only 2.

    Returns list of lists of strings.

    :param attribute_columns: list of strings containing all attribute columns
    
    """
    # if there are only 2 columns, return them both
    if len(attribute_columns)<3:
      return attribute_columns
    
    # pick 3 random columns
    cols = random.sample(attribute_columns, 3)
    
    # this creates a list like ["A&B", "B&C", "A&C"] - it's useful to have an AND statement in the training
    # columns, otherwise EM training tables balloon
    training_columns = ["&".join([cols[0], cols[1]]), "&".join([cols[1], cols[2]]), "&".join([cols[0], cols[2]])]
    
    return training_columns
  

  def train_linker(
    self,
    data: typing.Union[pyspark.sql.dataframe.DataFrame, list],
    space:dict,
    attribute_columns:list,
    unique_id:str,
    training_columns:list=None
    ) -> typing.Tuple[splink.spark.spark_linker.SparkLinker, splink.splink_dataframe.SplinkDataFrame]:
    """
    Method to train a linker model.

    Returns a 2-tuple of trained linker object and Splink dataframe with predictions

    :param data: Spark DataFrame containing the data to be de-duplicated
    :param space: dictionary generated by hyperopt sampling
    :param attribute_columns: list of strings containing all attribute columns
    :param unique_id: string with the name of the unique ID column
    :param training_columns: list of strings containing training columns
    
    """


    # Drop any intermediate tables for a clean run
    self._drop_intermediate_tables()
    
    # Set up parameters

    # Get blocking rules from hyperopt space
    blocking_rules_to_generate_predictions = space["blocking_rules"]
    # TODO: separate out to own method?
    blocking_rules_to_generate_predictions = blocking_rules_to_generate_predictions.split(" OR ") if " OR " in blocking_rules_to_generate_predictions else [blocking_rules_to_generate_predictions]
    
    # Create comparison list from hyperopt space
    comparisons = self._create_comparison_list(space)

    deterministic_rules = self.deterministic_columns.split(" OR ") if " OR " in self.deterministic_columns else [self.deterministic_columns]
    print(deterministic_rules)
    # if deterministic columns are not set, pick 2 at random from attribute columns
    # and save them as an attribute so that they can remain consistent between runs
    if not training_columns:
      training_columns = self._randomise_columns(attribute_columns)
      self.training_columns = training_columns
      
    training_rules = self._generate_rules(training_columns)

    # create settings dict
    # TODO: make em_convergence not static?
    settings = {
      "retain_intermediate_calculation_columns": True,
      "retain_matching_columns": True,
      "link_type": self.linker_mode,
      "unique_id_column_name": unique_id,
      "comparisons": comparisons,
      "em_convergence": 0.01,
      "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions
    }

    # Train linker model
    if self.linker_mode == "dedupe_only":
      linker = SparkLinker(data, spark=self.spark, database=self.schema, catalog=self.catalog)
    else:
      linker = SparkLinker(data, spark=self.spark, database=self.schema, catalog=self.catalog, input_table_aliases=["df_left", "df_right"])
    
    linker.load_settings(settings)
    linker._settings_obj._probability_two_random_records_match = 1/self.data_rowcount
    linker.estimate_u_using_random_sampling(target_rows=self.data_rowcount)
    for training_rule in training_rules:
      linker.estimate_parameters_using_expectation_maximisation(training_rule)

    # Make predictions
    predictions = linker.predict()


    return linker, predictions
  
  
  
  def evaluate_linker(
    self,
    data: typing.Union[pyspark.sql.dataframe.DataFrame, list],
    predictions:splink.splink_dataframe.SplinkDataFrame,
    threshold:float,
    attribute_columns:list,
    unique_id:str,
    linker:splink.spark.spark_linker.SparkLinker,
    true_label:str=None
    ) -> dict:
    """
    Method to evaluate predictions made by a trained linker model.

    Returns a dictionary of evaluation metrics.

    :param data: Spark Dataframe containing the original dataset (required to establish ground truth labels)
    :param predictions: Splink DataFrame as a result of calling linker.predict()
    :param threshold: float indicating the probability threshold above which a pair is considered a match
    :param attribute_columns: list of strings with valid column names to compare the entropies of
    :param unique_id: string with the name of the unique ID column
    :param linker: isntance of splink.SparkLinker, used for deduplication (clustering)
    :param true_label: The name of the column with true record ids, if exists (default None) - if not None, auto_link will attempt to calculate empirical scores

    """
    
    # Cluster records that are matched above threshold
    # get adjusted base
    if type(data) != list:
      k = max([data.groupBy(cn).count().count() for cn in attribute_columns])
    else:
      # use the larger dataframe as baseline
      df0_size = data[0].count()
      df1_size = data[1].count()
      if df0_size < df1_size:
        k = max([data[1].groupBy(cn).count().count() for cn in attribute_columns])
      else:
        k = max([data[0].groupBy(cn).count().count() for cn in attribute_columns])

    clusters = linker.cluster_pairwise_predictions_at_threshold(predictions, threshold_match_probability=threshold)
    df_clusters = clusters.as_spark_dataframe()
    
    # Calculate mean change in entropy
    evals = self._calculate_unsupervised_metrics(df_clusters, attribute_columns, base=k)

    # empirical scores
    if true_label:
      cluster_metrics = self.get_clustering_metrics(df_clusters, true_label)
      for k, v in cluster_metrics.items():
        evals.update({k:v})


      df_predictions = predictions.as_spark_dataframe()
      confusion_metrics = self.get_confusion_metrics(data, df_predictions, threshold, unique_id, true_label)

      for k, v in confusion_metrics.items():
        evals.update({k:v})


    return evals
  
  
  
  def train_and_evaluate_linker(
    self,
    data: typing.Union[pyspark.sql.dataframe.DataFrame, list],
    space:dict,
    attribute_columns:list,
    unique_id:str,
    deterministic_columns:list=None,
    training_columns:list=None,
    threshold:float=0.9,
    true_label:str=None
    ) -> typing.Tuple[
      splink.spark.spark_linker.SparkLinker,
      splink.splink_dataframe.SplinkDataFrame,
      dict,
      dict,
      str
    ]:
    """
    Method to train and evaluate a linker model in one go

    Returns a tuple of trained linker, predictions, metrics, parameters and mlflow run_id

    :param data: Spark DataFrame containing the data to be de-duplicated
    :param space: dictionary generated by hyperopt sampling
    :param attribute_columns: list of strings containing all attribute columns
    :param unique_id: string with the name of the unique ID column
    :param deterministic columns: list of strings containint columns to block on - if None, they will be generated automatically/randomly
    :param training_columns: list of strings containing training columns - if None, they will be generated automatically/randomly
    :param threshold: float indicating the probability threshold above which a pair is considered a match
    :param true_label: The name of the column with true record ids, if exists (default None) - if not None, auto_link will attempt to calculate empirical scores
    """

    #set start time for measuring training duration
    start = datetime.now()
    
    # Train model
    linker, predictions = self.train_linker(data, space, attribute_columns, unique_id, training_columns)

    end = datetime.now()

    duration = (end-start).seconds
    
    # Evaluate model
    evals = self.evaluate_linker(data, predictions, threshold, attribute_columns, unique_id, linker, true_label)

    with mlflow.start_run() as run:
      splink_mlflow.log_splink_model_to_mlflow(linker, "linker")
      mlflow.log_metrics(evals)
      mlflow.log_metric("training_duration", duration)
      params = space.copy()
      params["deterministic_columns"] = self.deterministic_columns
      params["training_columns"] = self.training_columns
      mlflow.log_dict(params, "model_parameters.json")
      
    
    run_id = run.info.run_id
    return linker, predictions, evals, params, run_id

  def auto_link(
    self,
    data: typing.Union[pyspark.sql.dataframe.DataFrame, list],
    attribute_columns:list=None,
    unique_id:str=None,
    comparison_size_limit:int=100000,
    max_evals:int=1,
    cleaning="all",
    threshold:float=0.9,
    true_label:str=None,
    random_seed:int=42,
    metric:str="information_gain_power_ratio"
  ) -> None:
    """
    Method to run a series of hyperopt trials.
    

    :param data: Either a Spark DataFrame containing the data to be de-duplicated, or a list of 2 Spark Dataframes to be linked.
    :param space: dictionary generated by hyperopt sampling
    :param attribute_columns: list of strings containing all attribute columns
    :param unique_id: string with the name of the unique ID column
    :param comparison_size_limit: int denoting maximum size of pairs allowed after blocking
    :param max_evals: int denoting max number of hyperopt trials to run
    :param cleaning: string ("all" or "none") or dictionary with keys as column names and values as list of strings for methods (accepted are "lower" and "alphanumeric_only")
    :param threshold: float indicating the probability threshold above which a pair is considered a match
    :param true_label: The name of the column with true record ids, if exists (default None) - if not None, auto_link will attempt to calculate empirical scores
    :param random_seed: Seed for Hyperopt fmin
    """

    self._evaluate_data_input_arg(data)
    self._autolink_data = data
    self._cleaning_mode = cleaning
    self.attribute_columns = attribute_columns
    self.unique_id = unique_id

    # set autolinker mode based on input data arg
    if type(self._autolink_data) == list:
      self.linker_mode = "link_only"
    else:
      self.linker_mode = "dedupe_only"

    # set spark paramaters if not provided
    if not self.spark:
      self.spark = self._get_spark(self._autolink_data)
    if not self.catalog:
      self.catalog = self._get_catalog(self.spark)
    if not self.schema:
      self.schema = self._get_schema(self.spark)
    # turn off AQE for duration of linking
    self.spark.conf.set("spark.databricks.optimizer.adaptive.enabled", 'False')

    # set mlflow details
    if not self.experiment_name:
      self.experiment_name = self._set_mlflow_experiment_name(self.spark)
    mlflow.set_experiment(self.experiment_name)

    # set attribute columns if not provided
    if not self.attribute_columns:
      self.attribute_columns, self._autolink_data = self._create_attribute_columns(self._autolink_data)


    # Count rows in data - doing this here so we only do it once
    self.data_rowcount = self._get_rowcounts(self.linker_mode, self._autolink_data)
    
    # Clean the data
    self._autolink_data = self._do_column_cleaning(
      self._autolink_data
      ,self.linker_mode
      ,self.attribute_columns
    )

    # set unique id if not provided
    if not self.unique_id:
      self._autolink_data = self._set_unique_id(self._autolink_data)
      self.unique_id = "unique_id"


    
    # define objective function
    def tune_model(space):
      linker, predictions, evals, params, run_id = self.train_and_evaluate_linker(
        self._autolink_data,
        space,
        self.attribute_columns,
        unique_id=self.unique_id,
        deterministic_columns=self.deterministic_columns,
        training_columns=self.training_columns,
        threshold=threshold,
        true_label=true_label
      )
      
      loss = -evals[metric]
      
      result = {'loss': loss, 'status': STATUS_OK, 'run_id': run_id}
      for k, v in evals.items():
        result.update({k:v})

      return result

    # initialise trials and create hyperopt space
    self.trials = Trials()
    if self.linker_mode == "dedupe_only":
      space = self._create_hyperopt_space(self._autolink_data, self.attribute_columns, comparison_size_limit)
    else:
      # use the larger dataframe as baseline
      df0_size = self._autolink_data[0].count()
      df1_size = self._autolink_data[1].count()
      if df0_size < df1_size:
        space = self._create_hyperopt_space(self._autolink_data[1], self.attribute_columns, comparison_size_limit)
      else:
        space = self._create_hyperopt_space(self._autolink_data[0], self.attribute_columns, comparison_size_limit)

      # run hyperopt trials
    self.best = fmin(
        fn=tune_model,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=self.trials,
        rstate=np.random.default_rng(random_seed)
      )
        
    # Get best params and retrain
    best_param_for_rt = self._convert_hyperopt_to_splink()
    
    self.best_run_id = self.trials.best_trial["result"]["run_id"]
    self.best_linker, self.best_predictions_df = self.train_linker(self._autolink_data, best_param_for_rt, self.attribute_columns, self.unique_id, self.training_columns)

    
    # return succes text
    success_text = f"""
    ======================================================================================
                                      AutoLinking completed.
    ======================================================================================
    """

    print(success_text)
    self.spark.conf.set("spark.databricks.optimizer.adaptive.enabled", 'True')
    return None

  def _get_spark(
          self
          ,autolink_data: typing.Union[pyspark.sql.dataframe.DataFrame, list]
  ):
    # extract spark from input data
    if type(autolink_data) == list:
      spark = autolink_data[0].sparkSession
    else:
      spark = autolink_data.sparkSession
    return spark
  def _get_catalog(self, spark):
    return spark.catalog.currentCatalog()

  def _get_schema(self, spark):
    return spark.catalog.currentDatabase()

  def _set_mlflow_experiment_name(self, spark):
    username = spark.sql('select current_user() as user').collect()[0]['user']
    experiment_name =  f"/Users/{username}/Databricks Autolinker {str(datetime.now())}"
    return experiment_name

  def _create_attribute_columns(
          self
          ,autolink_data
  ):
    """
    Called only when an autolink process is initiated, this function will calculate which attribute columns to use and
    do the necessary remapping work.
    Returns
    -------

    """
    data = autolink_data
    if type(data) == list:
      s1 = set(data[0].columns)
      s2 = set(data[1].columns)
      if s1 == s2:
        attribute_columns = data[0].columns
      else:
        # sort tables so the one with fewer columns is first.
        data.sort(key=lambda x: -len(x.columns))
        # do remappings
        remappings = self.estimate_linking_columns(data)
        data[0] = data[0].selectExpr("*", *[f"{x[0]} as {x[2]}" for x in remappings])
        data[1] = data[1].selectExpr("*", *[f"{x[1]} as {x[2]}" for x in remappings])
        # finally, set attribute columns
        attribute_columns = [x[2] for x in remappings]
    else:
      attribute_columns = self.estimate_clustering_columns(data)
    return attribute_columns, data

  def _get_rowcounts(self, linker_mode, autolink_data):
    if linker_mode == "dedupe_only":
      data_rowcount = autolink_data.count()
    else:
      # use the larger dataframe as baseline
      df0_size = autolink_data[0].count()
      df1_size = autolink_data[1].count()
      if df0_size < df1_size:
        data_rowcount = df1_size
      else:
        data_rowcount = df0_size
    return data_rowcount

  def _set_unique_id(self, autolink_data):
    if type(autolink_data) == list:
      autolink_data[0] = autolink_data[0].withColumn("unique_id", F.monotonically_increasing_id())
      autolink_data[1] = autolink_data[1].withColumn("unique_id", F.monotonically_increasing_id())
    else:
      autolink_data = autolink_data.withColumn("unique_id", F.monotonically_increasing_id())

    return autolink_data


  def _evaluate_data_input_arg(
          self,
          data: typing.Union[pyspark.sql.dataframe.DataFrame, list]
  ):
    # evaluate input argument of data
    _data_error_message = "The data argument accepts a single spark dataframe for deduplication, or a list of 2 " \
                          "spark dataframes for linking"
    if type(data) not in set([pyspark.sql.dataframe.DataFrame, list]):
      raise ValueError(_data_error_message)
    if type(data) is list:
      data_len = len(data)
      data_types = {type(x) for x in data}
      if data_len != 2:
        raise ValueError(_data_error_message)
      if len(data_types) != 1:
        raise ValueError(_data_error_message)
      if data_types.pop() != pyspark.sql.dataframe.DataFrame:
        raise ValueError(_data_error_message)

  def estimate_linking_columns(
          self
          ,data: list
  ):
    '''
    This function estimates the attribute columns for linking 2 datasets. It does this by joining each dataset on each
    column to every other column, and choosing the pairing with the highest count.

    Parameters
    ----------
    data : 2 spark dataframes which will be linked

    Returns
    - a mapping of the 2 tables to a standard schema.
    -------

    '''
    columns = [list(filter(lambda x: x != self.unique_id, x.columns)) for x in data]

    # write sql to lowercase and remove all non alpha-numeric characters
    cleaning_sql = 'lower(regexp_replace({column},  "[^0-9a-zA-Z]+", "")) as {column}'
    sqls = [list(map(lambda x: cleaning_sql.format(column=x), y)) for y in columns]

    # apply sql
    cleaned_data = [
      data[0].selectExpr(sqls[0]),
      data[1].selectExpr(sqls[1])
    ]

    # cross product of column names to get all the joins.
    # remove reversed duplicates (A,B) == (B, A)
    column_joins = itertools.product(*columns)
    final_joins = set()
    for x in column_joins:
      if (x[1], x[0]) not in final_joins:
        final_joins.add(x)

    # loop through the final joins and store the counts with the join criteria
    results = []
    for x in final_joins:
      results.append([*x, cleaned_data[0].join(cleaned_data[1], on=cleaned_data[0][x[0]]==cleaned_data[1][x[1]], how="inner").count()])
    # sort the results to get an ordered list of (cola, colb, count) ordered by cola and count descending
    results.sort(key=lambda x: (x[0], -x[2]))

    # find the column comparison with the highest count
    comparisons = [results[0]]
    for x in results[1:]:
      if x[0] == comparisons[-1][0]:
        pass
      else:
        comparisons.append(x)

    # create mapping of old names to new names
    updated_name_mappings = []
    for x in comparisons:
      updated_name_mappings.append([x[0], x[1], "".join([x[0], x[1]])])

    return updated_name_mappings

  def estimate_clustering_columns(
          self
          , data: pyspark.sql.dataframe.DataFrame
  ):
    '''
    Use all values as attributes for deduping.
    This method is in place in case we want to do something different in future.
    Parameters
    ----------
    data :

    Returns
    -------

    '''
    return data.columns

  def get_best_splink_model(self):
    """
    Get the underlying Splink model from the MLFlow Model Registry. Useful for advanced users wishing to access Splink's
    internal functionality not exposed as part of arc Autolinker.
    Returns
    -------
    A Splink model object.
    """
    loaded_model = mlflow.pyfunc.load_model(f"runs:/{self.best_run_id}/linker")
    unwrapped_model = loaded_model.unwrap_python_model()
    return unwrapped_model
  

  def best_predictions(self):
    """
    Get the predictions from the best linker model trained.
    Returns a spark dataframe of the pairwise predictions made by the autolinker model.
    """
    return self.best_predictions_df.as_spark_dataframe()
  

  def _do_clustering(self):
    #factor out the clustering call as used multiple times
    clusters = self.best_linker.cluster_pairwise_predictions_at_threshold(self.best_predictions_df, self.cluster_threshold)
    self.clusters = clusters
  

  def best_clusters_at_threshold(
    self,
    threshold:float=0.8
    ) -> pyspark.sql.DataFrame:
    """
    Convert the pairwise predictions to clusters using the connected components algorithm.

    Parameters
    :param threshold: default value=0.8.An optional parameter controlling the threshold at which records will be connected. Set it higher to produce clusters with greater precision, lower to produce clusters with greater recall.

    Returns a spark dataframe of the clustered input data with a new column cluster_id prepended.

    """
    # if self.clusters is empty run clustering and log the threshold
    if not self.clusters:
      self.cluster_threshold=threshold
      self._do_clustering()
    # enter this clause if rerunning clustering. 
    else:
      # if the provided threshold is the same as last time, no need to recalculate
      if threshold == self.cluster_threshold:
        pass
      else:
        self.cluster_threshold=threshold
        self._do_clustering()
    return self.clusters.as_spark_dataframe()
        
  
  def cluster_viewer(self):
    """
    Produce an interactive dashboard for visualising clusters. It provides examples of clusters of different sizes.
    The shape and size of clusters can be indicative of problems with record linkage, so it provides a tool to help you
    find potential false positive and negative links.
    See this video for an indepth explanation of interpreting this dashboard: https://www.youtube.com/watch?v=DNvCMqjipis

    Writes a HTML file to DBFS at "/dbfs/Users/{username}/scv.html"

    """
    # do clustering if not already done and no clusters provided
    if not self.clusters:
      raise ValueError("Pairs have not yet been clustered. Please run best_clusters_at_threshold() with an optional"
                       "threshold argument to generate clusters. ")
      
    path=f"/Users/{self.username}/clusters.html"
    self.best_linker.cluster_studio_dashboard(self.best_predictions_df, self.clusters, path, sampling_method="by_cluster_size", overwrite=True)
    
    # splink automatically prepends /dbfs to the output path
    with open("/dbfs" + path, "r") as f:
        html2=f.read()
    

    displayHTML(html2)
    
  # TODO: temporarily disabled because Sphinx doesn't play with displayHTML()    
  def comparison_viewer(self) -> None:
    """
    Produce an interactive dashboard for querying comparison details.
    See this video for an indepth explanation of interpreting this dashboard: https://www.youtube.com/watch?v=DNvCMqjipis

    Writes a HTML file to DBFS at "/dbfs/Users/{username}/scv.html"

    Returns None.
    """
    path=f"/dbfs/Users/{self.username}/scv.html"

    self.best_linker.comparison_viewer_dashboard(self.best_predictions_df, path, overwrite=True)

    with open("/dbfs" + path, "r") as f:
        html=f.read()
    
  
    displayHTML(html)
 

  def match_weights_chart(self) -> None:
    """
    Get the match_weights_chart

    """
    return self.best_linker.match_weights_chart()
    
    
  def get_scores_df(self, data_df, predictions_df, unique_id, true_label):
    """
    
    """
    left_df = data_df.select(F.col(unique_id).alias(f"{unique_id}_l"), F.col(true_label).alias("true_label"))
    right_df = data_df.select(F.col(unique_id).alias(f"{unique_id}_r"), F.col(true_label).alias("score_label"))
    
    return predictions_df\
      .select("match_probability", f"{unique_id}_l", f"{unique_id}_r")\
      .join(left_df, on=[f"{unique_id}_l"])\
      .join(right_df, on=[f"{unique_id}_r"])\
      .withColumnRenamed("match_probability", "probability")
    
  def get_RR_count(self, data_df, unique_id, true_label):
    left_df = data_df.select(F.col(true_label), F.col(unique_id).alias(f"{unique_id}_l"))
    right_df = data_df.select(F.col(true_label), F.col(unique_id).alias(f"{unique_id}_r"))
    
    pairs_df = left_df\
      .join(right_df, on=[true_label])\
      .where(f"{unique_id}_l != {unique_id}_r")
    
    unique_pairs_df = pairs_df\
      .withColumn("pairs", F.array(F.col(f"{unique_id}_l"), F.col(f"{unique_id}_r")))\
      .withColumn("pairs", F.array_sort("pairs"))
      
    return unique_pairs_df\
      .groupBy("pairs").count()\
      .count()
  
  def get_PR_count(self, scores_df, unique_id):
    unique_pairs_df = scores_df\
      .withColumn("pairs", F.array(F.col(f"{unique_id}_l"), F.col(f"{unique_id}_r")))\
      .withColumn("pairs", F.array_sort("pairs"))
    
    return unique_pairs_df\
      .groupBy("pairs").count()\
      .count()
  
  
  def get_confusion_metrics(self, data_df, predictions_df, thld, unique_id, true_label):
    
    scores_df = self.get_scores_df(data_df, predictions_df, unique_id, true_label)
  
    # RR - Relevant Records
    RR = self.get_RR_count(data_df, unique_id, true_label)
      
    calibrated_scores_df = scores_df.where((F.col("probability") > thld))

    FP = calibrated_scores_df.where(F.col("true_label") != F.col("score_label")).count()
    TP = calibrated_scores_df.where(F.col("true_label") == F.col("score_label")).count()
    # PR - Positive Records
    PR = self.get_PR_count(calibrated_scores_df, unique_id)

    if (PR > 0):
      precision = TP / PR
    else:
      precision = 0
      
    if (RR > 0):
      recall = TP / RR
    else:
      recall = 0

    if (precision+recall > 0):
      f1 = 2*precision*recall/(precision+recall)
    else:
      f1 = 0

    if (PR + FP)>0:
      jaccard = TP/(PR + FP)
    else:
      jaccard = 0

    scores = {
      "threshold": thld,
      "f1_score": f1,
      "precision": precision,
      "recall": recall,
      "jaccard": jaccard
      }

    return scores
  
  def get_clustering_metrics(self, clusters, true_label):
    win = Window.partitionBy(true_label)
    win2 = Window.partitionBy("cluster_id")

    pdf_clusters = clusters \
      .withColumn("cnt", F.count("*").over(win))\
      .withColumn(true_label, F.when(F.col("cnt") == 1, F.lit(-1)).otherwise(F.col(true_label)))\
      .withColumn("cnt", F.count("*").over(win2))\
      .withColumn("cluster_id", F.when(F.col("cnt") == 1, F.lit(-1)).otherwise(F.col("cluster_id")))\
      .toPandas()

    cluster_scores = {
      "adjusted_mutual_info_score": adjusted_mutual_info_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
      "adjusted_rand_score": adjusted_rand_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
      "completeness_score": completeness_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
      "fowlkes_mallows_score": fowlkes_mallows_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
      "homogeneity_score": homogeneity_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
      "mutual_info_score": mutual_info_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
      "normalized_mutual_info_score": normalized_mutual_info_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
      "rand_score": rand_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
      "v_measure_score": v_measure_score(pdf_clusters[true_label], pdf_clusters["cluster_id"]),
    }

    return cluster_scores

  def delete_all_linking_experiments(self):
    if input("WARNING - this will delete all your ARC generated MLFlow experiments, Type 'YES' to proceed" ) != "YES":
      return
    username = self.spark.sql('select current_user() as user').collect()[0]['user']
    pattern = f"%{username}/Databricks Autolinker%"

    client = MlflowClient()
    experiments = (
      client.search_experiments(filter_string=f"name LIKE '{pattern}'")
    )  # returns a list of mlflow.entities.Experiment
    for exp in experiments:
      client.delete_experiment(exp.experiment_id)