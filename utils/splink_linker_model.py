import mlflow
import mlflow.pyfunc
import os
import random
import splink.spark.spark_comparison_library as cl
import tempfile

from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score, f1_score
from splink.spark.spark_linker import SparkLinker

from utils.mlflow_utils import *


class SplinkLinkerModel(mlflow.pyfunc.PythonModel):
    """
    MLFlow compatibility wrapper for splink linker.
    """
 
    def __init__(self, **kwargs):
        self.settings = {}
        self.deterministic_rules = None
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __getstate__(self):
        json_result = self.get_settings_object()
        return json_result
            
    def __setstate__(self, json_dict):
        self.settings = json_dict
    
    def load_context(self, context):
        # this simply stores the json with the MLFLow model.
        self.context = context
        
    def clear_context(self):
        # model cannot be pickled, so disposing it
        self.linker.spark = None
        
    def set_should_evaluate(self, flag):
        self.should_evaluate = flag
        
    def spark_linker(self, data):
        spark = SparkSession.builder.getOrCreate()
        self.linker = SparkLinker(data, spark=spark)
        spark = None
        return self.linker
        
    def set_spark_linker(self, linker: SparkLinker):
        self.linker = linker
    
    def set_settings(self, settings):
        self.settings = settings
        if self.linker:
            self.linker.initialise_settings(self.settings)
        
    def set_blocking_rules(self, blocking_rules):
        self.settings.update({"blocking_rules_to_generate_predictions": blocking_rules})
        if self.linker:
            self.linker.initialise_settings(self.settings)
    
    def set_comparisons(self, comparisons):
        self.settings.update({"comparisons": comparisons})
        if self.linker:
            self.linker.initialise_settings(self.settings)
            
    def set_deterministic_rules(self, rules):
        self.deterministic_rules = rules
            
    def set_stage1_columns(self, columns):
        self.stage1_columns = columns
        
    def set_stage2_columns(self, columns):
        self.stage2_columns = columns
    
    def set_target_rows(self, rows):
        self.target_rows = rows
        
    def get_linker(self):
        return self.linker
    
    def get_settings(self):
        return self.settings
   
    def get_settings_object(self):
        return self.get_linker()._settings_obj.as_dict()
    
    def estimate_random_match_probability(self, rules, recall):
        self.linker.estimate_probability_two_random_records_match(rules, recall=recall)
        
    def estimate_u(self, target_rows):
        self.linker.estimate_u_using_random_sampling(target_rows=target_rows)
        
    def estimate_m(self, data, columns_1, columns_2):
        fixed_columns = random.choices(data.columns, k=2)
        training_rule_1 = " and ".join([f"(l.{cn} = r.{cn})" for cn in columns_1])
        training_rule_2 = " and ".join([f"(l.{cn} = r.{cn})" for cn in columns_2])
        self.linker.estimate_parameters_using_expectation_maximisation(training_rule_1)
        self.linker.estimate_parameters_using_expectation_maximisation(training_rule_2)
        
    def fit(self, X, recall_prior=0.8):
        """
        Model training given input parameters and a training dataset.
        """
        if self.settings is None:
            raise Exception("Cannot initialise linker without settings being set. Please use model.set_settings.")
        if not self.deterministic_rules:
            raise Exception("Cannot initialise linker without setting deterministic rules. Please use model.set_deterministic_rules.")
        
      
        linker = self.spark_linker(X)
        linker.initialise_settings(self.settings)
        self.set_spark_linker(linker)
        self.estimate_random_match_probability(self.deterministic_rules, recall_prior)
        self.estimate_u(self.target_rows)
        self.estimate_m(X, self.stage1_columns, self.stage2_columns)
        return linker
    
    def log_settings_as_json(self, path):
        """
        Simple method for logging a splink model
        Parameters
        ----------
        linker : Splink model object
        Returns
        -------
        """
        path = "linker.json"
        if os.path.isfile(path):
            os.remove(path)
        self.linker.save_settings_to_json(path)
        mlflow.log_artifact(path)
        os.remove(path)
        
    def _log_chart(self, chart_name, chart):
        '''
        Save a chart to MLFlow. This writes the chart out temporarily for MLFlow to pick up and save.
        Parameters
        ----------
        chart_name : str, the name the chart will be given in MLFlow
        chart : chart object from Splink
        Returns
        -------
        '''
        path = f"{chart_name}.html"
        if os.path.isfile(path):
            os.remove(path)
        save_offline_chart(chart.spec, path)
        mlflow.log_artifact(path)
        os.remove(path)
  
    def evaluate(self, y_test, y_test_pred):
        """
        Evaluate model performance.
        """
        if self.should_evaluate:
            spark = SparkSession.builder.getOrCreate()
            splink_tables = spark.sql('show tables like "*__splink__*"')
            temp_tables = splink_tables.collect()
            drop_tables = list(map(lambda x: x.tableName, temp_tables))
            for x in drop_tables:
                spark.sql(f"drop table {x}")
            self.linker.register_table(y_test, "labels")
            roc_auc = linker.roc_chart_from_labels_table("labels")
            return roc_auc
        else:
            return None
 
    def predict(self, context, X):
        """
        Predict labels on provided data
        """
        self.linker = self.spark_linker(X)
        self.linker.initialise_settings(self.settings)
        result = self.linker.predict()
        return result
    
