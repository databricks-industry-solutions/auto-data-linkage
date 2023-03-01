from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
from dbruntime.display import displayHTML

from splink.spark.spark_linker import SparkLinker
from splink.databricks.enable_splink import enable_splink
import splink.spark.spark_comparison_library as cl

import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from hyperopt.pyll import scope

from arc.autolinker import splink_mlflow

import numpy as np
import itertools
import math
import random
from datetime import datetime
from time import time

import mlflow


class AutoLinker:
  """
  Class to create object arc for automated data linking.
  
  Basic usage:
  
  >>> arc = AutoLinker(
  ...   spark=spark                              # Spark instance
  ...   catalog="splink_catalog",                # catalog name
  ...   schema="splink_schema",                  # schema to write results to
  ...   experiment_name="autosplink_experiment"  # MLflow experiment location
  ... )
  >>> best_linker = arc.auto_link(
  ...   data=data,                               # dataset to dedupe
  ...   attribute_columns=["A", "B", "C", "D"],  # columns that contain attributes to compare
  ...   unique_id="id",                          # column name of the unique ID
  ...   comparison_size_limit=500000,            # Maximum number of pairs when blocking applied
  ...   max_evals=100                            # Maximum number of hyperopt trials to run
  ... )
  >>> best_linker.predict()                      # use Splink functionality out-of-the-box

  """
  
  
  def __init__(self, spark=None, catalog=None, schema=None, experiment_name=None, training_columns=None, deterministic_columns=None):

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
    self.df_true_positives=None

  def __str__(self):
    return f"AutoLinker instance working in {self.catalog}.{self.schema} and MLflow experiment {experiment_name}"
  
  
  
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
    # will always be between 0 and 1
    vc = data.groupBy(column).count().withColumn("norm_count", F.col("count")/self.data_rowcount)

    
    # Calculate P*ln(P) per row. Milos idea: e^P+ln(ln(P)) 
    # will always be  <= 0
    vc = vc.withColumn("entropy", F.col("norm_count")*F.log(F.col("norm_count")))
    
    # Entropy = -SUM(P*ln(P))
    # will always be >=0
    entropy = -vc.select(F.sum(F.col("entropy")).alias("sum_entropy")).collect()[0].sum_entropy
    
    # make it bigger to get away from zero
    entropy = np.exp(entropy)
    
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
  
  
  
  def _generate_candidate_blocking_rules(self, data, attribute_columns, comparison_size_limit, unique_id, max_columns_per_rule=2):
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
      blocking_combinations.extend(combs)
      
    
    # Convert them to Splink-compatible rules
    blocking_rules = self._generate_rules(["&".join(r) for r in blocking_combinations])
    
    # Estimate number of pairs per blocking rule by grouping on blocking rule columns and getting the sum of squares of counts
    comp_size_dict = dict()

    for comb, rule in zip(blocking_combinations, blocking_rules):
      num_pairs = data.groupBy(list(comb)).count().select((F.sum(F.col("count")*(F.col("count")-F.lit(1))))/F.lit(2)).collect()[0]["(sum((count * (count - 1))) / 2)"]
      comp_size_dict.update({rule: num_pairs})
    
    # we want all rule sets to contain at least 2 different rules - otherwise we end up with only a single blocking rule
    potential_rules = []
    for r in range(2, len(blocking_rules)+1):
      for c in itertools.combinations(blocking_rules, r):
        rule_size = sum([v for k, v in comp_size_dict.items() if k in c])
        potential_rules.append(list([c, rule_size]))

    # take the bottom 4 rules by size if no comparison size is returned
    if not comparison_size_limit:
      potential_rules.sort(key=lambda x: x[-1])
      accepted_rules = potential_rules[:4]
      accepted_rules = [_[0] for _ in accepted_rules]
      
    else:
      # loop through all combinations of combinations and save those which remain under the limit (this is a bit brute force)
      accepted_rules = []
      for rule in potential_rules:
        if rule[1]<=comparison_size_limit:
          accepted_rules.append(rule[0])


    if len(accepted_rules)<1:
      min_size = min(comp_size_dict.values())
      raise ValueError(f"No blocking rules meet the comparison size limit. Minimum pair space = {min_size}")
          
    return accepted_rules

  
  
  def _create_hyperopt_space(self, data, attribute_columns, comparison_size_limit, unique_id, max_columns_per_rule=2):
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
    # Drop intermediate tables in schema for consecutive runs
    tables_in_schema = self.spark.sql(f"show tables from {self.catalog}.{self.schema} like '*__splink__*'").collect()
    for table in tables_in_schema:
      try:
        self.spark.sql(f"drop table {self.catalog}.{self.schema}.{table.tableName}") 
      except:
        self.spark.sql(f"drop table {table.tableName}")
        
  
  def _clean_columns(self, data, attribute_columns, cleaning):
    """
    Method to clean string columns (turn them to lower case and remove non-alphanumeric characters)
    in order to help with better (and quicker) string-distance calculations. If cleaning is "all"
    (as is by default), it will automatically clean as much as it can.  If cleaning is "none", it will do nothing.
    cleaning can also be a dictionary with keys as column names and values as lists of method strings.
    The currently available cleaning methods are turning to lower case and removing non-alphanumeric characters.
    Parameters
    : data : Spark DataFrame containing the data to be cleaned
    : attribute_columns : list of strings containing all attribute columns
    : cleaning : string or dicitonary with keys as column names and values as list of valid cleaning method names.
    Returns
      - Spark dataframe
    """
    
    # if cleaning is set to "none", don't do anything to the data
    if cleaning=="none":
      return data
    
    # if it's set to "all", turn it into a dictionary covering all possible cases
    if cleaning=="all":
      cleaning = {col: ["lower", "alphanumeric_only"] for col in attribute_columns}
      
    for col, methods in cleaning.items():
      # if column is not a string, skip it
      if not data.schema[col].dataType.typeName == "string":
        continue
        
      for method in methods:
        if method=="alphanumeric_only":
          # replace column and only keep alphanumeric and space characters
          data = data.withColumn(col, F.regexp_replace(F.col(col), r"[^A-Za-z0-9 ]+", ""))
        
        elif method=="lower":
          data = data.withColumn(col, F.lower(F.col(col)))
          
    return data
  


  def _convert_hyperopt_to_splink(self):
    """
    Method to convert hyperopt trials to a dictionary that can be used to
    train a linker model. Used for training the best linker model at the end of an experiment.
    Sets class attributes for best metric and parameters as well
    Returns
      - dictionary
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
    Method to deduplicate values within the original dataset given the predictions dataframe, its linker
    object and an optional threshold. Cluster values are standardised to a single value.
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
    # define window to aggregate over
    window = Window.partitionBy("cluster_id")
    df_predictions = clusters.as_spark_dataframe()
    # loop through the attributes and standardise. we want to keep original column names.
    for attribute_column in attribute_columns:
      new_col_name = f"std_{attribute_column}"
      df_predictions = (
        df_predictions
        .withColumn(new_col_name, F.first(F.col(attribute_column)).over(window)) # standardise values
        .drop(F.col(attribute_column)) # drop origina values
        .withColumnRenamed(new_col_name, attribute_column) # replace with standardised values
      )
      
    return df_predictions

  

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
  
  
  
  def _randomise_columns(self, attribute_columns):
    """
    Method to randomly select (combinations of) columns from attribute columns for EM training.
    Will try to pick 2 combinations of 2 (i.e AB and BC from ABC), but will default to 2 if there are only 2.
    Parameters
    : attribute_columns : list of strings containing all attribute columns
    Returns
      - List of lists of strings
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

    
    # Deduplicate data
    deduped = self.deduplicate_records(data, predictions, linker, attribute_columns, unique_id, threshold)
    
    # Calculate mean change in entropy
    mean_entropy_change = self.calculate_entropy_delta(data, deduped, attribute_columns)
    
    evals = {
      "mean_entropy_change": mean_entropy_change
    }

    return evals
  
  
  
  def train_and_evaluate_linker(self
                                , data
                                , space
                                , attribute_columns
                                , unique_id
                                , deterministic_columns=None
                                , training_columns=None
                                , threshold=0.9
                                , true_id=None):
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
      - 7-tuple of trained linker (SparkLinker instance), predictions (Splink DAtaFrame) and metrics (4x float) and mlflow run_id
    """

    #set start time for measuring training duration
    start = datetime.now()
    
    # Train model
    linker, predictions = self.train_linker(data, space, attribute_columns, unique_id, deterministic_columns, training_columns)

    end = datetime.now()

    duration = (end-start).seconds
    
    # Evaluate model
    evals = self.evaluate_linker(data, predictions, threshold, attribute_columns, unique_id, linker)
    
    if true_id:
      precision, recall, f1 = self._calculate_empirical_score(data, predictions, unique_id, threshold, true_id)
      ground_truth_metrics = {
          "precision":precision
        , "recall":recall
        , "f1":f1
      }
       

    with mlflow.start_run() as run:
      splink_mlflow.log_splink_model_to_mlflow(linker, "linker")
      mlflow.log_metrics(evals)
      mlflow.log_metric("training_duration", duration)
      params = space.copy()
      params["deterministic_columns"] = self.deterministic_columns
      params["training_columns"] = self.training_columns
      mlflow.log_dict(params, "model_parameters.json")
      mlflow.log_metric("run_end_time", time())
      if true_id:
        mlflow.log_metrics(ground_truth_metrics)
      
    
    run_id = run.info.run_id
    return linker, predictions, evals, params, run_id

  
  
  def auto_link(self
                , data
                , attribute_columns
                , unique_id, max_evals
                , comparison_size_limit=None
                , cleaning="all"
                , deterministic_columns=None
                , training_columns=None
                , threshold=0.9
                , true_id=None
               ):
    """
    Method to run a series of hyperopt trials.
    Parameters
    : data : Spark DataFrame containing the data to be de-duplicated
    : space : dictionary generated by hyperopt sampling
    : attribute_columns : list of strings containing all attribute columns
    : unique_id : string with the name of the unique ID column
    : comparison_size_limit : int denoting maximum size of pairs allowed after blocking
    : max_evals : int denoting max number of hyperopt trials to run
    : cleaning : string ("all" or "none") or dictionary with keys as column names and values as list of strings for methods (accepted are "lower" and "alphanumeric_only")
    : deterministic columns : list of strings containint columns to block on - if None, they will be generated automatically/randomly
    : training_columns : list of strings containing training columns - if None, they will be generated automatically/randomly
    : threshold : float indicating the probability threshold above which a pair is considered a match
    Returns
      - splink.SparkLinker instance (best linker)
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
        true_id=true_id
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
  
  def best_clusters_at_threshold(self, threshold=0.8):
    """
    Convert the pairwise predictions to clusters using the connected components algorithm.

    Parameters
    ----------
    threshold : default value=0.8.An optional parameter controlling the threshold at which records will be connected.
                Set it higher to produce clusters with greater precision, lower to produce clusters with greater recall.

    Returns
    -------
    A spark dataframe of the clustered input data with a new column cluster_id prepended.

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

    Returns
    -------

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
    
    
  def comparison_viewer(self):
    """
    Produce an interactive dashboard for querying comparison details.
    See this video for an indepth explanation of interpreting this dashboard: https://www.youtube.com/watch?v=DNvCMqjipis

    Writes a HTML file to DBFS at "/dbfs/Users/{username}/scv.html"

    Returns None.
    -------

    """
    path=f"/dbfs/Users/{self.username}/scv.html"

    self.best_linker.comparison_viewer_dashboard(self.best_predictions_df, path, overwrite=True)

    with open("/dbfs" + path, "r") as f:
        html=f.read()

    displayHTML(html)
 

  def match_weights_chart(self):
    """
    Get the
    Returns
    -------

    """
    return self.best_linker.match_weights_chart()
  
  
  def _calculate_empirical_score(self, data, predictions, unique_id, threshold, true_id):
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
    data = data.withColumn("recid", F.col(true_id))
    if not self.df_true_positives:
      # filter original data to only contain rows where recid id appears more than once (known dupes)
      data_recid_groupby = data.groupBy("recid").count().filter("count>1").withColumnRenamed("recid", "recid_")
      data_tp = data.join(data_recid_groupby, data.recid==data_recid_groupby.recid_, how="inner").drop("recid_")

      data_l = data_tp.withColumnRenamed(unique_id, f"{unique_id}_l").withColumnRenamed("recid", "recid_l").select(f"{unique_id}_l", "recid_l")
      data_r = data_tp.withColumnRenamed(unique_id, f"{unique_id}_r").withColumnRenamed("recid", "recid_r").select(f"{unique_id}_r", "recid_r")

      dt = data_l.join(data_r, data_l[f"{unique_id}_l"]!=data_r[f"{unique_id}_r"], how="inner")
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
      ((df_pred[f"{unique_id}_l"]==self.df_true_positives[f"{unique_id}_r"]) & (df_pred[f"{unique_id}_l"]==self.df_true_positives[f"{unique_id}_r"])) | ((df_pred[f"{unique_id}_l"]==self.df_true_positives[f"{unique_id}_r"]) & (df_pred[f"{unique_id}_r"]==self.df_true_positives[f"{unique_id}_l"])),
      how="inner"
    ).count()
    
    # FN is the left anti-join of true and predicted
    fn = self.df_true_positives.join(
      df_pred,
      ((df_pred[f"{unique_id}_l"]==self.df_true_positives[f"{unique_id}_l"]) & (df_pred[f"{unique_id}_r"]==self.df_true_positives[f"{unique_id}_r"])) | ((df_pred[f"{unique_id}_l"]==self.df_true_positives[f"{unique_id}_r"]) & (df_pred[f"{unique_id}_r"]==self.df_true_positives[f"{unique_id}_l"])),
      how="left_anti"
    ).count()
    
    # FP is the left anti-join of predicted and true
    fp = df_pred.join(
      self.df_true_positives,
      ((df_pred[f"{unique_id}_l"]==self.df_true_positives[f"{unique_id}_l"]) & (df_pred[f"{unique_id}_r"]==self.df_true_positives[f"{unique_id}_r"])) | ((df_pred[f"{unique_id}_l"]==self.df_true_positives[f"{unique_id}_r"]) & (df_pred[f"{unique_id}_r"]==self.df_true_positives[f"{unique_id}_l"])),
      how="left_anti"
    ).count()
    
    # TN is everything else, i.e. N(N-1)-TP-FN-FP
    N = data.count()
    tn = (N*(N-1))-tp-fn-fp
    
    # Calculate precision, recall and f1
    if tp+fp>0.0:
      precision = tp/(tp+fp)
    else:
      precision: 0.9
      
    if tp+fn>0.0:
      recall = tp/(tp+fn)
    else:
      recall = 0.0

    if (precision+recall)>0:
      f1 = precision*recall/(precision+recall)
    else:
      f1 = 0.0
      
    return precision, recall, f1
  
    
    
