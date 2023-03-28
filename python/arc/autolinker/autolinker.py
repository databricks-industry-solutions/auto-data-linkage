from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark
from arc.sql import functions as arcf

# from dbruntime.display import displayHTML

import splink
from splink.spark.spark_linker import SparkLinker
from splink.databricks.enable_splink import enable_splink
import splink.spark.spark_comparison_library as cl

import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from hyperopt.pyll import scope

from . import splink_mlflow

import pandas as pd
import numpy as np
import scipy.stats as ss
import itertools
import math
import random
from datetime import datetime

import typing

import mlflow


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
    self.catalog = catalog# if catalog else spark.catalog.currentCatalog()
    self.schema = schema  #. if schema else spark.catalog.currentDatabase()
    
    #self.username = spark.sql('select current_user() as user').collect()[0]['user']
    self.experiment_name = experiment_name #if experiment_name else f"/Users/{self.username}/Databricks Autolinker {str(datetime.now())}"
    self.training_columns = training_columns
    self.deterministic_columns = deterministic_columns
    self.best_params = None
    self.clusters=None
    self.cluster_threshold=None
    self.original_entropies = dict()


  def __str__(self):
    return f"AutoLinker instance working in {self.catalog}.{self.schema} and MLflow experiment {self.experiment_name}"
  
  
  
  def _calculate_column_entropy(
    self,
    data: pyspark.sql.DataFrame,
    column: str,
    by_cluster: bool=False
    ) -> float:
    """
    Method to calculate column entropy given a dataset, based on:
    :math:`Entropy = -SUM(P*ln(P))`
    where P is the normalised count of the occurrence of each unique value in the column.
    
    :param data: input dataframe with row per record
    :param column: (valid) name of column to calculate entropy on
    :param by_cluster: if True, it will calculate the average entropy when the dataset is split by clusters.
    """
    
    if by_cluster:
      cluster_groupby = "cluster_id"
    else:
      cluster_groupby = F.lit(1)
    
    # replace null values with dummy
    data = data.fillna("null_")

    df_entropy = data \
      .groupBy(cluster_groupby) \
      .agg(arcf.arc_entropy_agg(column)).select(F.col(f"arc_entropyaggexpression({column}).{column}").alias(f"entropy"))
    
    mean_entropy = df_entropy.select(F.mean("entropy").alias("mean_entropy")).collect()[0]["mean_entropy"]
    
    return mean_entropy

  
  def _calculate_dataset_entropy(
      self,
      data:pyspark.sql.DataFrame,
      attribute_columns:list,
      by_cluster:bool=False
      ) -> float:
    """
    Method to calculate the average entropy of each attribute column in a dataset.

    Returns a float.

    :param data: input dataframe with row per record
    :param column: (valid) name of column to calculate entropy on
    :param by_cluster: if True, it will calculate the average entropy when the dataset is split by clusters.
    """
    if by_cluster:
      cluster_groupby = "cluster_id"
    else:
      cluster_groupby = F.lit(1)


    entropy = data \
      .fillna("null_") \
      .groupBy(cluster_groupby) \
      .agg(
        arcf.arc_entropy_agg(*attribute_columns).alias("ent_map")
      ) \
      .withColumn("entropy", F.explode(F.map_values("ent_map"))) \
      .groupBy(
        cluster_groupby
      ).agg(
        F.sum("entropy").alias("entropy")
      ) \
      .select(F.avg("entropy").alias("mean_entropy"))
    
    entropy = entropy.collect()[0]["mean_entropy"]

    if entropy is None:
      entropy = 0.0
    
    return entropy
  

  def _calculate_information_gain(
      self,
      clusters:pyspark.sql.DataFrame,
      attribute_columns:list
      ) -> float:
    """
    Method to calculate the information gain within clusters when the subset of the dataset that has been matched is split by clusters,
    i.e. entropy(matched data) - average(entropy(data within clusters)).

    Returns information gain metric, as float.

    :param clusters: dataframe returned from linker clustering
    :param attribute_columns: list of valid column names containing attributes in the clusters dataset
    """
    # Calculate count of rows in each cluster and rejoin
    cluster_counts = clusters.groupBy("cluster_id").count().withColumnRenamed("count", "_cluster_count")
    data = clusters.join(cluster_counts, on=["cluster_id"], how="left")
    
    # Filter on rows which have a match
    df_matched = data.filter("_cluster_count>1")

    # Calculate matched whole data entropy
    entropy_matched = self._calculate_dataset_entropy(df_matched, attribute_columns, by_cluster=False)
    
    # Calculate mean entropy by clusters
    entropy_cluster_mean = self._calculate_dataset_entropy(df_matched, attribute_columns, by_cluster=True)
    
    # Information gain: matched data entropy - mean entropy in each cluster
    information_gain = entropy_matched - entropy_cluster_mean
    
    return information_gain


    
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
    unique_id:str,
    max_columns_per_and_rule:int=2,
    max_rules_per_or_rule:int=3,
  ) -> list:
    """
    Method to automatically generate a list of lists of blocking rules to test, given a user-defined limit for
    pair-wise comparison size.

    Returns a nested list of lists with Splink-compatible blocking rule queries.

    :param data: input data with record-per-row
    :param attribute_columns: valid column names of data, containing all possible columns to block on
    :param comparison_size_limit: the maximum number of pairs we want to compare, to limit hardware issues
    :param unique_id: the name of the unique ID column in data
    :param max_columns_per_and_rule: the maximum number of column comparisons in a single rule to try
    :param max_rules_per_or_rule: the maximum number of rules comparisons in a composite rule to try

    """

    # replace null values with dummy
    data = data.fillna("null_")

    # Create blocking rules and size esimates
    df_rules = arcf.arc_generate_blocking_rules(data, max_columns_per_and_rule, max_rules_per_or_rule, *attribute_columns)

    # Filter on max
    rules = df_rules.filter(f"rule_squared_count > {comparison_size_limit}").collect()

    accepted_rules = [rule.splink_rule for rule in rules]

    # set deterministic rules to be 500th largest (or largest) blocking rule
    self.deterministic_columns = df_rules.orderBy(F.col("rule_squared_count")).limit(500).orderBy(F.col("rule_squared_count").desc()).limit(1).collect()[0]["splink_rule"]

          
    return accepted_rules

  
  
  def _create_hyperopt_space(
    self,
    data:pyspark.sql.DataFrame,
    attribute_columns:list,
    comparison_size_limit:int,
    unique_id:str,
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
      unique_id=unique_id,
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
    attribute_columns:list,
    cleaning="all"):
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
  


  def _convert_hyperopt_to_splink(self) -> dict:
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
  
  

  def deduplicate_records(
    self,
    predictions:splink.splink_dataframe.SplinkDataFrame,
    linker:splink.spark.spark_linker.SparkLinker,
    attribute_columns:list,
    threshold:float=0.9
  ) -> pyspark.sql.DataFrame:
    """
    Method to deduplicate the original dataset given the predictions dataframe, its linker
    object and an optional threshold. The clusters are grouped on and duplicates are set to the same value (arbitrarily).
    These are then replaced in the original dataset.
    
    Returns a spark Dataframe with deduplicated records.

    :param predictions: Splink Dataframe that's a result of a linker.predict() call
    :param linker: a trained linker object, needed to cluster
    :param attribute_columns: list of column names containing attribtues in the data
    :param threshold: the probability threshold above which a pair is considered a match
    
    """
    
    # Cluster records that are matched above threshold
    clusters = linker.cluster_pairwise_predictions_at_threshold(predictions, threshold_match_probability=threshold)
    # define window to aggregate over
    window = Window.partitionBy("cluster_id")
    df_predictions = clusters.as_spark_dataframe()
    # loop through the attributes and standardise. we want to keep original column names.
    for attribute_column in attribute_columns:
      df_predictions = (
        df_predictions
        .withColumn(attribute_column, F.first(F.col(attribute_column)).over(window)) # standardise values
      )

    return df_predictions
  
  
  
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
    data:pyspark.sql.DataFrame,
    space:dict,
    attribute_columns:list,
    unique_id:str,
    deterministic_columns:list=None,
    training_columns:list=None
    ) -> typing.Tuple[splink.spark.spark_linker.SparkLinker, splink.splink_dataframe.SplinkDataFrame]:
    """
    Method to train a linker model.

    Returns a 2-tuple of trained linker object and Splink dataframe with predictions

    :param data: Spark DataFrame containing the data to be de-duplicated
    :param space: dictionary generated by hyperopt sampling
    :param attribute_columns: list of strings containing all attribute columns
    :param unique_id: string with the name of the unique ID column
    :param deterministic columns: list of strings containint columns to block on
    :param training_columns: list of strings containing training columns
    
    """


    # Drop any intermediate tables for a clean run
    self._drop_intermediate_tables()
    
    # Set up parameters

    # Get blocking rules from hyperopt space
    blocking_rules_to_generate_predictions = space["blocking_rules"]
    # TODO: remove this once fixed the other end
    blocking_rules_to_generate_predictions = blocking_rules_to_generate_predictions.split(" OR ") if " OR " in blocking_rules_to_generate_predictions else [blocking_rules_to_generate_predictions]
    
    # Create comparison list from hyperopt space
    comparisons = self._create_comparison_list(space)

    # if deterministic columns are not set, pick 2 at random from attribute columns
    # and save them as an attribute so that they can remain consistent between runs
    # if not self.deterministic_columns:
    #   deterministic_columns = random.sample(attribute_columns, 2)
    #   self.deterministic_columns = deterministic_columns

    # deterministic_rules = self._generate_rules(deterministic_columns)
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
      "link_type": "dedupe_only",
      "unique_id_column_name": unique_id,
      "comparisons": comparisons,
      "em_convergence": 0.01,
      "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions
    }
    
    # Train linker model
    linker = SparkLinker(data, spark=self.spark, database=self.schema, catalog=self.catalog)
    
    linker.initialise_settings(settings)
    # linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.8)
    linker._settings_obj._probability_two_random_records_match = 1/self.data_rowcount
    linker.estimate_u_using_random_sampling(target_rows=self.data_rowcount)
    for training_rule in training_rules:
      linker.estimate_parameters_using_expectation_maximisation(training_rule)

    # Make predictions
    predictions = linker.predict()

    return linker, predictions
  
  
  
  def evaluate_linker(
    self,
    data:pyspark.sql.DataFrame,
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
    :param df_predictions: Spark DataFrame containing pair-wise predictions made by a linker model
    :param threshold: float indicating the probability threshold above which a pair is considered a match
    :param attribute_columns: list of strings with valid column names to compare the entropies of
    :param linker: isntance of splink.SparkLinker, used for deduplication (clustering)
    :param true_label: The name of the column with true record ids, if exists (default None) - if not None, auto_link will attempt to calculate empirical scores

    """

    
     # Cluster records that are matched above threshold
    clusters = linker.cluster_pairwise_predictions_at_threshold(predictions, threshold_match_probability=threshold)
    df_clusters = clusters.as_spark_dataframe()
    
    # Calculate mean change in entropy
    information_gain = self._calculate_information_gain(df_clusters, attribute_columns)

    # empirical scores      
    evals = dict()

    if true_label:
      df_predictions = predictions.as_spark_dataframe()
      scores = self.get_confusion_metrics(data, df_predictions, threshold, unique_id, true_label)

      for k, v in scores.items():
        evals.update({k:v})

    evals["information_gain"] = information_gain


    return evals
  
  
  
  def train_and_evaluate_linker(
    self,
    data:pyspark.sql.DataFrame,
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
    linker, predictions = self.train_linker(data, space, attribute_columns, unique_id, deterministic_columns, training_columns)

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
    data:pyspark.sql.DataFrame,
    attribute_columns:list,
    unique_id:str,
    comparison_size_limit:int,
    max_evals:int,
    cleaning="all",
    deterministic_columns:list=None,
    training_columns:list=None,
    threshold:float=0.9,
    true_label:str=None,
    random_seed:int=42
  ) -> None:
    """
    Method to run a series of hyperopt trials.
    

    :param data: Spark DataFrame containing the data to be de-duplicated
    :param space: dictionary generated by hyperopt sampling
    :param attribute_columns: list of strings containing all attribute columns
    :param unique_id: string with the name of the unique ID column
    :param comparison_size_limit: int denoting maximum size of pairs allowed after blocking
    :param max_evals: int denoting max number of hyperopt trials to run
    :param cleaning: string ("all" or "none") or dictionary with keys as column names and values as list of strings for methods (accepted are "lower" and "alphanumeric_only")
    :param deterministic columns: list of strings containint columns to block on - if None, they will be generated automatically/randomly
    :param training_columns: list of strings containing training columns - if None, they will be generated automatically/randomly
    :param threshold: float indicating the probability threshold above which a pair is considered a match
    :param true_label: The name of the column with true record ids, if exists (default None) - if not None, auto_link will attempt to calculate empirical scores
    
    """
    
    # extract spark from input data

    self.spark = data.sparkSession if not self.spark else self.spark
    self.catalog = self.catalog if self.catalog else self.spark.catalog.currentCatalog()
    self.schema = self.schema   if self.schema else self.spark.catalog.currentDatabase()
    
    self.username = self.spark.sql('select current_user() as user').collect()[0]['user']
    self.experiment_name = self.experiment_name if self.experiment_name else f"/Users/{self.username}/Databricks Autolinker {str(datetime.now())}"
    self.best_params = None
    
    mlflow.set_experiment(self.experiment_name)

    # Count rows in data - doing this here so we only do it once
    self.data_rowcount = data.count()
    
    # Clean the data
    data = self._clean_columns(data, attribute_columns, cleaning)
    
    # define objective function
    def tune_model(space):
      linker, predictions, evals, params, run_id = self.train_and_evaluate_linker(
        data,
        space,
        attribute_columns,
        unique_id=unique_id,
        deterministic_columns=self.deterministic_columns,
        training_columns=self.training_columns,
        threshold=threshold,
        true_label=true_label
      )
      
      loss = -evals["information_gain"]
      
      result = {'loss': loss, 'status': STATUS_OK, 'run_id': run_id}
      for k, v in evals.items():
        result.update({k:v})

      return result
    
    # initialise trials and create hyperopt space
    self.trials = Trials()
    space = self._create_hyperopt_space(data, attribute_columns, comparison_size_limit, unique_id)
    
      # run hyperopt trials
    self.best = fmin(
        fn=tune_model,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=self.trials,
        rstate=np.random.default_rng(random_seed)
      )
    
    best_param_for_rt = self._convert_hyperopt_to_splink()
    
    
    self.best_run_id = self.trials.best_trial["result"]["run_id"]
    self.best_linker, self.best_predictions_df = self.train_linker(data, best_param_for_rt, attribute_columns, unique_id, self.deterministic_columns, self.training_columns)
        
    
    # return succes text
    success_text = f"""
    ======================================================================================
                                      AutoLinking completed.
    ======================================================================================
    Deterministic rules used
    ------------------------
    {', '.join(self._generate_rules(self.deterministic_columns))}
    ------------------------
    Training rules used
    ------------------------
    {', '.join(self._generate_rules(self.training_columns))}
    ------------------------
    Best parameters
    ------------------------
    {self.best_params}
    ------------------------
    Best Metric: {self.best_metric}
    ------------------------
    You can now access the best linker model and predictions and use Splink's built-in functionality, e.g.:
    
    >>> linker = arc.best_linker
    >>> predictions = arc.best_predictions()
    >>> ...
    """
    print(success_text)
    
    return None
  
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
    Returns
    -------
    A spark dataframe of the pairwise predictions made by the autolinker model.
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
    
    ## TODO: this method currently breaks Sphinx, need to investigate
    displayHTML(html2)
    
    
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
    
    ## TODO: this method currently breaks Sphinx, need to investigate
    displayHTML(html)
 

  def match_weights_chart(self) -> None:
    """
    Get the
    Returns

    """
    return self.best_linker.match_weights_chart()
    
    
  def get_scores_df(self, data_df, predictions_df, unique_id, true_label):
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
