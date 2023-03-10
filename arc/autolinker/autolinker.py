from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window
import pyspark

# from dbruntime.display import displayHTML

import splink
from splink.spark.spark_linker import SparkLinker
from splink.databricks.enable_splink import enable_splink
import splink.spark.spark_comparison_library as cl

import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from hyperopt.pyll import scope

from . import splink_mlflow

import numpy as np
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
    column: str
  ) -> float:
    """
    Method to calculate column entropy given a dataset, based on:
    :math:`Entropy = -SUM(P*ln(P))`
    where P is the normalised count of the occurrence of each unique value in the column.
    
    :param data: input dataframe with row per record
    :param column: (valid) name of column to calculate entropy on

    """
    
    # calculate normalised value count per unique value in column
    rowcount = data.count()
    vc = data.groupBy(column).count().withColumn("norm_count", F.col("count")/rowcount)

    
    # Calculate P*ln(P) per row
    vc = vc.withColumn("entropy", F.col("norm_count")*F.log(F.col("norm_count")))
    
    # Entropy = -SUM(P*ln(P))
    entropy = -vc.select(F.sum(F.col("entropy")).alias("sum_entropy")).collect()[0].sum_entropy

    
    return entropy
  
  
    
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
    max_columns_per_rule:int=2
  ) -> list:
    """
    Method to automatically generate a list of lists of blocking rules to test, given a user-defined limit for
    pair-wise comparison size.

    Returns a nested list of lists with Splink-compatible blocking rule queries.

    :param data: input data with record-per-row
    :param attribute_columns: valid column names of data, containing all possible columns to block on
    :param comparison_size_limit: the maximum number of pairs we want to compare, to limit hardware issues
    :param unique_id: the name of the unique ID column in data
    :param max_columns_per_rule: the maximum number of column comparisons in a single rule to try

    """

    # initialise empty list to store all combinations in
    blocking_combinations = []

    # create list of all possible rules
    for n in range(1, max_columns_per_rule+1):
      combs = list(itertools.combinations(attribute_columns, n))
      blocking_combinations.extend(combs)

    # Convert them to Splink-compatible rules
    blocking_rules = self._generate_rules(["&".join(r) for r in blocking_combinations])
    
    # Estimate number of pairs per blocking rule by grouping on blocking rule columns and getting the sum of squares of counts
    comp_size_dict = dict()

    for comb, rule in zip(blocking_combinations, blocking_rules):
      num_pairs = data.groupBy(list(comb)).count().select(F.sum(F.col("count")*F.col("count"))).collect()[0]["sum((count * count))"]
      comp_size_dict.update({rule: num_pairs})
    
    # loop through all combinations of combinations and save those which remain under the limit (this is a bit brute force)
    accepted_rules = []
    for r in range(1, len(blocking_rules)+1):
      for c in itertools.combinations(blocking_rules, r):
        if sum([v for k, v in comp_size_dict.items() if k in c])<=comparison_size_limit:
          accepted_rules.append(list(c))


    if len(accepted_rules)<1:
      raise ValueError("No blocking rules meet the comparison size limit.")
          
    return accepted_rules

  
  
  def _create_hyperopt_space(
    self,
    data:pyspark.sql.DataFrame,
    attribute_columns:list,
    comparison_size_limit:int,
    unique_id:str,
    max_columns_per_rule:int=2
  ) -> dict:
    """
    Method to create hyperopt space for comparison and blocking rule hyperparameters from list of columns.
    Takes a given (or generated) list of columns and produces a dictionary that can be converted to a comparison list later on,
    with function names and threhsolds, and a list (of lists) of blocking rules to test.

    Returns a dictionary for hyperopt parameter search

    :param data: input data with records per row
    :param attribute_columns: valid column names of data, containing all possible columns to compare
    :param comparison_size_limit: maximum number of pairs we want to compare, to limit hardware issues
    :param max_columns_per_rule: the maximum number of column comparisons in a single rule to try
    
    """

    # Generate candidate blocking rules
    self.blocking_rules = self._generate_candidate_blocking_rules(
      data=data,
      attribute_columns=attribute_columns,
      comparison_size_limit=comparison_size_limit,
      unique_id=unique_id,
      max_columns_per_rule=max_columns_per_rule
    )

    # Create comparison dictionary using Hyperopt sampling
    space = dict()
    comparison_space = dict()

    # Create hyperopt space for comparisons
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

  

  def calculate_entropy_delta(
    self,
    deduped:pyspark.sql.DataFrame
  ) -> float:
    """
    Method to calculate the change in entropy of a set of columns between two
    datasets, used to calculate the information gain after deduping a set of records.
    
    Returns average change in column entropy.

    :param data: the original (undeduplicated) data set of records
    :param deduped: the new (deduplicated) data set of records
    :param columns: valid column names to compare
    
    """
    
    
    # Initialise empty list to populate with entropy deltas
    entropy_deltas = list()
    dd_cnt=deduped.count()
    columns = self.attribute_columns
    if self.original_entropies == dict():
      for column in columns:
        self.original_entropies[column] = self._calculate_column_entropy(self.data, column, self.data_rowcount)
    
    # For each column in given columns, calculate entropy in original dataset
    # and the deduplicated dataset. The delta is the difference between new and old.
    for column in columns:
      original_entropy = self.original_entropies[column]
      new_entropy = self._calculate_column_entropy(deduped, column, dd_cnt)
      entropy_deltas.append(new_entropy-original_entropy)


    return np.mean(entropy_deltas)
  
  
  
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
    blocking_rules_to_generate_predictions = list(space["blocking_rules"])
    print(blocking_rules_to_generate_predictions)
    # Create comparison list from hyperopt space
    comparisons = self._create_comparison_list(space)

    # if deterministic columns are not set, pick 2 at random from attribute columns
    # and save them as an attribute so that they can remain consistent between runs
    if not self.deterministic_columns:
      deterministic_columns = random.sample(attribute_columns, 2)
      self.deterministic_columns = deterministic_columns

    deterministic_rules = self._generate_rules(deterministic_columns)

    # if deterministic columns are not set, pick 2 at random from attribute columns
    # and save them as an attribute so that they can remain consistent between runs
    if not training_columns:
      training_columns = self._randomise_columns(attribute_columns)
      self.training_columns = training_columns
      
    training_rules = self._generate_rules(training_columns)

    # create settings dict
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
    linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.8)
    linker.estimate_u_using_random_sampling(target_rows=1e7)
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
    linker:splink.spark.spark_linker.SparkLinker
    ) -> dict:
    """
    Method to evaluate predictions made by a trained linker model.

    Returns a dictionary of evaluation metrics.

    :param data: Spark Dataframe containing the original dataset (required to establish ground truth labels)
    :param df_predictions: Spark DataFrame containing pair-wise predictions made by a linker model
    :param threshold: float indicating the probability threshold above which a pair is considered a match
    :param attribute_columns: list of strings with valid column names to compare the entropies of
    :param linker: isntance of splink.SparkLinker, used for deduplication (clustering)
    
    """

    
    # Deduplicate data
    deduped = self.deduplicate_records(data, predictions, linker, attribute_columns, unique_id, threshold)
    
    # Calculate mean change in entropy
    mean_entropy_change = self.calculate_entropy_delta(data, deduped, attribute_columns)
    
    evals = {
      "mean_entropy_change": mean_entropy_change
    }

    return evals
  
  
  
  def train_and_evaluate_linker(
    self,
    data:pyspark.sql.DataFrame,
    space:dict,
    attribute_columns:list,
    unique_id:str,
    deterministic_columns:list=None,
    training_columns:list=None,
    threshold:float=0.9
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
    
    """

    #set start time for measuring training duration
    start = datetime.now()
    
    # Train model
    linker, predictions = self.train_linker(data, space, attribute_columns, unique_id, deterministic_columns, training_columns)

    end = datetime.now()

    duration = (end-start).seconds
    
    # Evaluate model
    evals = self.evaluate_linker(data, predictions, threshold, attribute_columns, unique_id, linker)

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
    threshold:float=0.9
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
        threshold=threshold
      )
      
      loss = evals["mean_entropy_change"]
      
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
        trials=self.trials
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

    displayHTML(html)
 

  def match_weights_chart(self) -> None:
    """
    Get the
    Returns

    """
    return self.best_linker.match_weights_chart()
    
    
