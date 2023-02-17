# Databricks notebook source
# MAGIC %pip install --quiet splink mlflow hyperopt

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

from splink.spark.spark_linker import SparkLinker
from splink.databricks.enable_splink import enable_splink
import splink.spark.spark_comparison_library as cl

import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.mixture import GaussianMixture

from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pandas as pd
import numpy as np
import itertools
import math
import random
from datetime import datetime
from pprint import pprint

import mlflow

# COMMAND ----------

class AutoLinker:
  """
  Class to create object autolinker for automated data linking
  """
  
  def __init__(self, schema, experiment_name):
#     self.spark = spark
    self.schema = schema
    self.experiment_name = experiment_name
    self.training_columns = None
    self.deterministic_columns = None
    self.df_true_positives = None
    
    spark.sql(f"USE {schema}")
    mlflow.set_experiment(experiment_name)

  
  
  def _calculate_column_entropy(self, data, column):
    """
    Method to calculate column entropy given a dataset, based on:
    Entropy = -SUM(P*ln(P))
    where P is the normalised count of the occurrence of each unique value in the column
    Parameters
    : data : Spark Dataframe
    : column : name of column to calculate
    Returns
      - Float
    """
    
    # calculate normalised value count per unique value in colunn
    rowcount = data.count()
    vc = data.groupBy(column).count().withColumn("norm_count", F.col("count")/rowcount)
    
#     _udf_entropy = F.udf(lambda x: self._calculate_p(x), DoubleType())
    
    # Calculate P*ln(P) per row
    vc = vc.withColumn("entropy", F.col("norm_count")*F.log(F.col("norm_count")))
    
    # Entropy = -SUM(P*ln(P))
    entropy = -vc.select(F.sum(F.col("entropy")).alias("sum_entropy")).collect()[0].sum_entropy

    
    return entropy
  
    
  def _generate_rules(self, columns):
    """
    Method to create a list of Splink-compatible SQL statements from a list of candidate columns for rules.
    Can be used for simple blocking rules (no AND statement), deterministic rules or for training rules
    Parameters
    : columns : list of str containing valid columns in the data set
    Returns
      - List of strings of Splink-compatible SQL statements
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
  
  
  def _generate_candidate_blocking_rules(self, data, attribute_columns, comparison_size_limit, max_columns_per_rule=2):
    """
    Method to automatically generate a list of lists of blocking rules to test, given a user-defined limit for
    pair-wise comparison size.
    Parameters
    : data : Spark DataFrame containing all records to de-duplicate
    : attribute_columns : list of strings with valid column names of data, containing all possible columns to block on
    : comparison_size_limit : integer denoting the maximum number of pairs we want to compare - to limit hardware issues
    : max_columns_per_rule : integer denoting the maximum number of column comparisons in a single rule to try
    Returns
      - nested list of lists with Splink-compatible blocking rule queries
    """

    # initialise empty list to store all combinations in
    blocking_combinations = []

    # create list of all possible rules
    for n in range(1, max_columns_per_rule+1):
      combs = list(itertools.combinations(attribute_columns, n))

      rules = ["&".join(comb) for comb in combs]
      blocking_combinations.extend(rules)

    # generate Splink-compatible blocking rules
    blocking_rules_to_generate_predictions = self._generate_rules(blocking_combinations)

    # create dummy linker to count number of combinations
    linker = SparkLinker(data, spark=spark)
    settings = {
      "retain_intermediate_calculation_columns": True,
      "retain_matching_columns": True,
      "link_type": "dedupe_only",
      "unique_id_column_name": "uid",
      "em_convergence": 0.01,
      "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions
    }
    linker.initialise_settings(settings)

    # create dictionary of all possible combinations and their sizes
    comp_size = [linker.count_num_comparisons_from_blocking_rule(i) for i in blocking_rules_to_generate_predictions]
    comp_size_dict = {r: s for r, s in zip(blocking_rules_to_generate_predictions, comp_size)}

    # loop through all combinations of combinations and save those which remain under the limit (this is a bit brute force)
    accepted_rules = []
    for r in range(1, len(blocking_rules_to_generate_predictions)+1):
      for c in itertools.combinations(blocking_rules_to_generate_predictions, r):
        if sum([v for k, v in comp_size_dict.items() if k in c])<=comparison_size_limit:
          accepted_rules.append(list(c))

    # clean up intermediate tables (not sure if needed)
    x = spark.sql("show tables like '*__splink__*'").collect()
    for _ in x:
        spark.sql(f"drop table {_.tableName}") 

    return accepted_rules

  
  def _create_hyperopt_space(self, data, attribute_columns, comparison_size_limit, max_columns_per_rule=2):
    """
    Method to create hyperopt space for comparison and blocking rule hyperparameters from list of columns.
    Takes a given (or generated) list of columns and produces a dictionary that can be converted to a comparison list later on,
    with function names and threhsolds, and a list (of lists) of blocking rules to test.
    Parameters
    : data : Spark DataFrame containing all records to de-duplicate
    : attribute_columns : list of strings with valid column names of data, containing all possible columns to block on or compare
    : comparison_size_limit : integer denoting the maximum number of pairs we want to compare - to limit hardware issues
    : max_columns_per_rule : integer denoting the maximum number of column comparisons in a single rule to try
    Returns
      - hyperopt.pyll graph for hyperopt parameter search
    """

    # Generate candidate blocking rules
    self.blocking_rules = self._generate_candidate_blocking_rules(
      data=data,
      attribute_columns=attribute_columns,
      comparison_size_limit=comparison_size_limit,
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
    # Drop intermediate tables in schema for consecutive runs
    x = spark.sql("show tables like '*__splink__*'").collect()
    for _ in x:
      spark.sql(f"drop table {_.tableName}") 
        
  
  def _create_comparison_list(self, space):
    """
    Method to convert comparisons dictionary generated by hyperopt to a Splink-compatible list of
    spark_comparison_library functions with the generated columns and thresholds.
    Parameters
    : comparison_dict : nested dicitonary generated by create_comparison_dict
    Returns
      - List of spark_comparison_library method instances
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
  

  def deduplicate_records(self, data, predictions, linker, attribute_columns, unique_id, threshold=0.9):
    """
    Method to deduplicate the original dataset given the predictions dataframe, its linker
    object and an optional threshold. The clusters are grouped on and duplicates are removed (arbitrarily).
    These are then replaced in the original dataset.
    Parameters
    : data : Spark Dataframe of the original data set
    : predictions : Splink Dataframe that's a result of a linker.predict() call
    : linker : the linker object (splink.SparkLinker instance), needed to cluster
    : threshold : Float denoting the probability threshold above which a pair is considered a match
    Returns
      - Spark Dataframe with deduplicated records
    """
    
    # Cluster records that are matched above threshold
    clusters = linker.cluster_pairwise_predictions_at_threshold(predictions, threshold_match_probability=threshold)
    df_predictions = clusters.as_spark_dataframe()

    ids_to_agg = attribute_columns
    ids_to_agg.append(unique_id)
    
    # Deduplicate predictions by grouping on cluster_id
    df_deduped_preds = df_predictions.groupBy("cluster_id").agg(
      *(F.first(i).alias(i) for i in ids_to_agg)
    )

    # Left anti-join predictions on original data to get all records outside of blocking rules
    data_without_predictions = data.join(
      df_predictions,
      data.uid==df_predictions.uid,
      how="left_anti"
    )

    # Union deduped predictions and data outside of blocking rules
    df_deduped = df_deduped_preds \
      .select(ids_to_agg) \
      .union(
        data_without_predictions \
          .select(ids_to_agg)
    )

    return df_deduped


  def calculate_entropy_delta(self, data, deduped, columns):
    """
    Method to calculate the change in entropy of a set of columns between two
    datasets, used to calculate the information gain after deduping a set of records.
    Parameters
    : data : Spark DataFrame, the original (undeduplicated) data set of records
    : deduped : Spark DataFrame, the new (deduplicated) data set of records
    : columns : list of strings with valid column names to compare
    Returns
      - Float (average change in column entropy)
    """
    
    # Initialise empty list to populate with entropy deltas
    entropy_deltas = list()
    
    # For each column in given columns, calculate entropy in original dataset
    # and the deduplicated dataset. The delta is the difference between new and old.
    for column in columns:
      original_entropy = self._calculate_column_entropy(data, column)
      new_entropy = self._calculate_column_entropy(deduped, column)
      entropy_deltas.append(new_entropy-original_entropy)


    return np.mean(entropy_deltas)
  
  
  
  def calculate_empirical_score(self, data, predictions, threshold):
    """
    Method to calculate precision, recall and F1 score based on ground truth labels.
    Assumes the data has a column with empirical IDs to assess whether two records belong
    to the same real life entity. Will check if this has already been calculated. If not,
    it will create it for the first (and only) time.
    NB: this includes a bit of hard coding, but it won't make it to production anyway because
    we won't have ground truth.
    Parameters
    : data : Spark DataFrame containing the data to be de-duplicated
    : predictions : Splink DataFrame with predicted pairs
    : threshold : float indicating the probability threshold above which a pair is considered a match
    Returns
      - 3-tuple of floats for precision, recall and F1 score
    """
    if not self.df_true_positives:
      # filter original data to only contain rows where recid id appears more than once (known dupes)
      data_recid_groupby = data.groupBy("recid").count().filter("count>1").withColumnRenamed("recid", "recid_")
      data_tp = data.join(data_recid_groupby, data.recid==data_recid_groupby.recid_, how="inner").drop("recid_")

      data_l = data_tp.withColumnRenamed("uid", "uid_l").withColumnRenamed("recid", "recid_l").select("uid_l", "recid_l")
      data_r = data_tp.withColumnRenamed("uid", "uid_r").withColumnRenamed("recid", "recid_r").select("uid_r", "recid_r")

      dt = data_l.join(data_r, data_l.uid_l!=data_r.uid_r, how="inner")
      # create boolean col for y_true
      df_true = dt.withColumn("match", F.when(F.col("recid_l")==F.col("recid_r"), 1).otherwise(0))
      # only keep matches
      df_true = df_true.filter("match=1")
      # assign as attribute to avoid re-calculation
      self.df_true_positives = df_true
      
    # convert predictions to Spark DataFrame and filter on match prob threshold - table will only contain predicted positives
    df_pred = predictions.as_spark_dataframe().filter(f"match_probability>={threshold}")
    
    # Calculate TP, FP, FN, TN
    
    # TP is the inner join of true and predicted pairs
    tp = df_pred.join(
      self.df_true_positives,
      ((df_pred.uid_l==self.df_true_positives.uid_l) & (df_pred.uid_r==self.df_true_positives.uid_r)) | ((df_pred.uid_l==self.df_true_positives.uid_r) & (df_pred.uid_r==self.df_true_positives.uid_l)),
      how="inner"
    ).count()
    
    # FN is the left anti-join of true and predicted
    fn = self.df_true_positives.join(
      df_pred,
      ((df_pred.uid_l==self.df_true_positives.uid_l) & (df_pred.uid_r==self.df_true_positives.uid_r)) | ((df_pred.uid_l==self.df_true_positives.uid_r) & (df_pred.uid_r==self.df_true_positives.uid_l)),
      how="left_anti"
    ).count()
    
    # FP is the left anti-join of predicted and true
    fp = df_pred.join(
      self.df_true_positives,
      ((df_pred.uid_l==self.df_true_positives.uid_l) & (df_pred.uid_r==self.df_true_positives.uid_r)) | ((df_pred.uid_l==self.df_true_positives.uid_r) & (df_pred.uid_r==self.df_true_positives.uid_l)),
      how="left_anti"
    ).count()
    
    # TN is everything else, i.e. N(N-1)-TP-FN-FP
    N = data.count()
    tn = (N*(N-1))-tp-fn-fp
    
    # Calculate precision, recall and f1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    if (precision+recall)>0:
      f1 = precision*recall/(precision+recall)
    else:
      f1 = 0.0
      
    return precision, recall, f1

  def train_linker(self, data, space, attribute_columns, unique_id, deterministic_columns=None, training_columns=None):
    """
    Method to train a linker model
    Parameters
    : data : Spark DataFrame containing the data to be de-duplicated
    : space : dictionary generated by hyperopt sampling
    : attribute_columns : list of strings containing all attribute columns
    : unique_id : string with the name of the unique ID column
    : deterministic columns : list of strings containint columns to block on
    : training_columns : list of strings containing training columns
    Returns
      - tuple of trained linker object and Splink dataframe with predictions
    """

    # Drop any intermediate tables for a clean run
    self._drop_intermediate_tables()
    
    # Set up parameters

    # Get blocking rules from hyperopt space
    blocking_rules_to_generate_predictions = list(space["blocking_rules"])

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
      training_columns = random.sample(attribute_columns, 2)
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
    linker = SparkLinker(data, spark=spark)
    linker.initialise_settings(settings)
    linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.8)
    linker.estimate_u_using_random_sampling(target_rows=1e7)
    for training_rule in training_rules:
      linker.estimate_parameters_using_expectation_maximisation(training_rule)

    # Make predictions
    predictions = linker.predict()


    return linker, predictions
  
  def evaluate_linker(self, data, predictions, threshold, attribute_columns, unique_id, linker):
    """
    Method to evaluate predictions made by a trained linker model
    Parameters
    : data : Spark Dataframe containing the original dataset (required to establish ground truth labels)
    : df_predictions : Spark DataFrame containing pair-wise predictions made by a linker model
    : threshold : float indicating the probability threshold above which a pair is considered a match
    : attribute_columns : list of strings with valid column names to compare the entropies of
    : linker : isntance of splink.SparkLinker, used for deduplication (clustering)
    Returns
      - Dictionary of evaluation metrics TODO: make the metrics controllable via an argument
    """

    # Calculate empirical scores
    precision, recall, f1 = self.calculate_empirical_score(data, predictions, threshold)
    
    # Deduplicate data
    deduped = self.deduplicate_records(data, predictions, linker, attribute_columns, unique_id, threshold)
    
    # Calculate mean change in entropy
    mean_entropy_change = self.calculate_entropy_delta(data, deduped, attribute_columns)
    
    evals = {
      "precision": precision,
      "recall": recall,
      "f1": f1,
      "mean_entropy_change": mean_entropy_change
    }

    return evals
  
  
  def train_and_evaluate_linker(self, data, space, attribute_columns, unique_id, deterministic_columns=None, training_columns=None, threshold=0.9):
    """
    Method to train and evaluate a linker model in one go
    Parameters
    : data : Spark DataFrame containing the data to be de-duplicated
    : space : dictionary generated by hyperopt sampling
    : attribute_columns : list of strings containing all attribute columns
    : unique_id : string with the name of the unique ID column
    : deterministic columns : list of strings containint columns to block on - if None, they will be generated automatically/randomly
    : training_columns : list of strings containing training columns - if None, they will be generated automatically/randomly
    : threshold : float indicating the probability threshold above which a pair is considered a match
    Returns
      - 6-tuple of trained linker (SparkLinker instance), predictions (Splink DAtaFrame) and metrics (4x float)
    """

    # Train model
    linker, predictions = self.train_linker(data, space, attribute_columns, unique_id, deterministic_columns, training_columns)

    # Evaluate model
    evals = self.evaluate_linker(data, predictions, threshold, attribute_columns, unique_id, linker)


    return linker, predictions, evals
  


  def log_to_mlflow(self, trials):
    """
    Method to log parameters and loss function of each trial as an MLflow run.
    Parameters
    : trials : hyperopt.Trials object created after a hyperopt run
    Returns:
      - None
    """
    
    for t in trials.trials:
      with mlflow.start_run():
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

        params = {
          "blocking_rules": self.blocking_rules[blocking_rules],
          "comparisons": comparison_params,
          "deterministic_rules": self._generate_rules(self.deterministic_columns),
          "training_rules": self._generate_rules(self.training_columns),
        }
        metrics = {k:v for k, v in t["result"].items() if not k=="status"}
        mlflow.log_dict(params, "params.json")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    return None
  
  
  def auto_link(self, data, attribute_columns, unique_id, comparison_size_limit, max_evals, deterministic_columns=None, training_columns=None, threshold=0.9):
    """
    Method to run a series of hyperopt trials.
    Parameters
    : data : Spark DataFrame containing the data to be de-duplicated
    : space : dictionary generated by hyperopt sampling
    : attribute_columns : list of strings containing all attribute columns
    : unique_id : string with the name of the unique ID column
    : comparison_size_limit : int denoting maximum size of pairs allowed after blocking
    : max_evals : int denoting max number of hyperopt trials to run
    : deterministic columns : list of strings containint columns to block on - if None, they will be generated automatically/randomly
    : training_columns : list of strings containing training columns - if None, they will be generated automatically/randomly
    : threshold : float indicating the probability threshold above which a pair is considered a match
    Returns
      - None
    """
    
    #set start time for measuring duration
    start = datetime.now()
    
    # define objective function
    def tune_model(space):
      linker, predictions, evals = self.train_and_evaluate_linker(
        data,
        space,
        attribute_columns,
        unique_id="uid",
        deterministic_columns=self.deterministic_columns,
        training_columns=self.training_columns,
        threshold=threshold
      )
      
      loss = evals["mean_entropy_change"]
      
      result = {'loss': loss, 'status': STATUS_OK}
      for k, v in evals.items():
        result.update({k:v})

      return result
    
    # initialise trials and create hyperopt space
    self.trials = Trials()
    space = self._create_hyperopt_space(data, attribute_columns, comparison_size_limit)

    # run hyperopt trials
    self.best = fmin(
      fn=tune_model,
      space=space,
      algo=tpe.suggest,
      max_evals=max_evals,
      trials=self.trials
    )
    
    # log trials to MLflow

    self.log_to_mlflow(self.trials)
    
    # Set best params, metrics and results
    t = self.trials.best_trial
    self.best_metric = -t['result']['loss']
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
    
    # TODO: move this into its own method, probably - refactor...
    # convert best params to a format that can retrain a new linker 
    # model based on the best params. This is necessary because the underlying tables may have been overwritten
    best_comparisons = {}
    for k, v in comparison_params.items():
      best_comparisons.update({
        k: {"distance_function": {"distance_function": v['function'], "threshold": v['threshold']}}
      })
      
    best_params_for_rt = {
      "blocking_rules": self.best_params['blocking_rules'],
      "comparisons": best_comparisons
    }
    
    self.best_linker, self.best_predictions = self.train_linker(data, best_params_for_rt, attribute_columns, unique_id, self.deterministic_columns, self.training_columns)
    
    #end training duration
    end = datetime.now()
    training_duration = (end-start).seconds
    
    # return succes text
    success_text = f"""
    ======================================================================================
    AutoLinking completed. Carried out {max_evals} trials in {training_duration} seconds.
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
    Best F1: {self.best_metric}
    ------------------------
    
    You can now access the best linker model and predictions and use Splink's built-in functionality, e.g.:
    
    >>> linker = autolinker.best_linker
    >>> predictions = autolinker.best_predictions
    >>> ...
    """
    
    print(success_text)
    

# COMMAND ----------

# MAGIC %md
# MAGIC # North Carolina Voters dataset
# MAGIC https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

# COMMAND ----------

data = spark.read.table("marcell_autosplink.voters_data_sample")

# COMMAND ----------

autolinker = AutoLinker(
  #spark=spark,                                                                                               # spark instance
  schema="marcell_autosplink",                                                                               # schema to write results to
  experiment_name="/Users/marcell.ferencz@databricks.com/autosplink/evaluate/autosplink_experiment_large"  # MLflow experiment location
)

# COMMAND ----------

autolinker.auto_link(
  data=data,                                                         # dataset to dedupe
  attribute_columns=["givenname", "surname", "suburb", "postcode"],  # columns that contain attributes to compare
  unique_id="uid",                                                   # column name of the unique ID
  comparison_size_limit=200000,                                      # Maximum number of pairs when blocking applied
  max_evals=60                                                      # Maximum number of hyperopt trials to run
)

# COMMAND ----------

linker = autolinker.best_linker

# COMMAND ----------

linker.m_u_parameters_chart()

# COMMAND ----------

predictions = autolinker.best_predictions

predictions.as_spark_dataframe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Music albums dataset
# MAGIC https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution

# COMMAND ----------

data = spark.read.table("marcell_autosplink.music_data").sample(0.5)

# COMMAND ----------

data = data \
  .withColumn("title", F.substring(F.col("title"), 0, 20)) \
  .withColumn("artist", F.substring(F.col("artist"), 0, 20)) \
  .withColumn("album", F.substring(F.col("album"), 0, 20))

# COMMAND ----------

autolinker = AutoLinker(
  #spark=spark,                                                                                               # spark instance
  schema="marcell_autosplink",                                                                               # schema to write results to
  experiment_name="/Users/marcell.ferencz@databricks.com/autosplink/evaluate/autosplink_music_experiment"  # MLflow experiment location
)

# COMMAND ----------

autolinker.auto_link(
  data=data,                                                         # dataset to dedupe
  attribute_columns=["title", "length", "artist", "album", "year", "language"],  # columns that contain attributes to compare
  unique_id="uid",                                                   # column name of the unique ID
  comparison_size_limit=200000,                                      # Maximum number of pairs when blocking applied
  max_evals=100                                                      # Maximum number of hyperopt trials to run
)
