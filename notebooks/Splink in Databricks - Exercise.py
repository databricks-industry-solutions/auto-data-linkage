# Databricks notebook source
# MAGIC %md
# MAGIC # Splink on Databricks - Exercise

# COMMAND ----------

# MAGIC %pip install splink --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Libraries

# COMMAND ----------

from splink.spark.spark_linker import SparkLinker
from splink.databricks.enable_splink import enable_splink
import splink.spark.spark_comparison_library as cl

import pandas as pd

from utils.splink_linker_model import SplinkLinkerModel
from utils.mlflow_utils import *

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC Set parameters based on active user

# COMMAND ----------

username = spark.sql('select current_user() as user').collect()[0]['user']
db_name = username.replace(".", "_").replace("@", "_")
db_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating Splink
# MAGIC 
# MAGIC Whilst Splink does offer a supervised predictive method, our data does not have ground truth (we don't categorically know which records are duplicates or otherwise), so we're resigned to taking an unsupervised approach.
# MAGIC 
# MAGIC Unsupervised models are trickier to evaluate, because we can't compare the predictions against a known truth. How would we know how well our model has performed then? How can we evaluate the impact of changing parameters against the previous experiment?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Devising an evaluation metric
# MAGIC 
# MAGIC We need a metric that can compare the predictions of different models against one another. We know that the Splink model creates a pairwise dataframe (within our defined comparison space) and each pair is assigned a probability between 0 and 1.
# MAGIC 
# MAGIC Suppose that we have predicted 10,000 pairs of records' probability of matching and we have three models that made these predicitons:

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 1: The Perfect model
# MAGIC 
# MAGIC Our first model is perfect, therefore it predicted the pairs to be either 100% a match \\(Pr(match)=1\\) or 100% not a match \\(Pr(match)=0\\), so the distribution of the predicted probabilities looks something like the chart below.
# MAGIC 
# MAGIC We would want to score this model as high as possible (or give it a loss as low as possible).

# COMMAND ----------

test_data_perfect = pd.DataFrame({
  "match_probability": np.append(np.repeat([0.0], 5000), np.repeat([1.0], 5000))
})

test_data_perfect["match_probability"].hist(bins=50)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 2: The Worst model
# MAGIC 
# MAGIC What would be the worst case scenario? If the model was unsure about _every_ pair, it might predict everything to be 50%, i.e. \\(Pr(match)=0.5\\), so the distribution would look something like the chart below.
# MAGIC 
# MAGIC We would want to score this model low (or give it a high loss).

# COMMAND ----------

test_data_worst = pd.DataFrame({
  "match_probability": np.repeat([0.5], 10000)
})

test_data_worst["match_probability"].hist(bins=50)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 3: A realistically good model
# MAGIC 
# MAGIC What would our predictions looks like in reality? Probably some minor % of predicitons distributed near the 100% probability mark (as we expect there to be fewer matches than not), and the majority distributed near the 0% probability mark.
# MAGIC 
# MAGIC We would want to score this higher than the worst and lower than the perfect.

# COMMAND ----------

test_data_realistic = pd.DataFrame({
  "match_probability": np.array([0.0 if x<=0.0 else 1.0 if x>=1.0 else x for x in np.append(np.random.normal(0.1, 0.01, 9500), np.random.normal(0.9, 0.01, 500))])
})

test_data_realistic["match_probability"].hist(bins=50)

# COMMAND ----------

# MAGIC %md
# MAGIC #### The proposed loss metric
# MAGIC 
# MAGIC Since we know what our edge cases (and a case somewhere in the middle) look like, we can attempt to assign a metric to these probability distributions:
# MAGIC 
# MAGIC 1. We fit 2 normal distributions to the predictions (since we expect two peaks around 0 and 1) with a Gaussian Mixture model
# MAGIC 2. We estimate the mean and the standard deviations of these two distributions
# MAGIC 3. Our loss function is then the distance of the distributions' means from 0 and 1, respectively, and the size of their standard deviations, i.e.:
# MAGIC 
# MAGIC \\(\mathcal{L}=\mu_{1}+\sigma_1+(1-\mu_{2})+\sigma_{2}\\)
# MAGIC 
# MAGIC Intuitively, the further away the average positive or negative prediction is from 0 and 1, and the wider the spread of the positive and negative predictions, the higher the loss, and the worse our model.
# MAGIC 
# MAGIC We have implemented this loss function in the `get_match_probability_loss` method, which we imported above.
# MAGIC 
# MAGIC Let's see how this would work on our test data:

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 1 Loss

# COMMAND ----------

loss_perfect, fig_perfect = get_match_probabilty_loss(test_data_perfect)
print(f"Loss for perfect scenario: {loss_perfect}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 2 Loss

# COMMAND ----------

loss_worst, fig_worst = get_match_probabilty_loss(test_data_worst)
print(f"Loss for worst scenario: {loss_worst}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model 3 Loss

# COMMAND ----------

loss_realistic, fig_realistic = get_match_probabilty_loss(test_data_realistic)
print(f"Loss for realistic scenario: {loss_realistic}")

# COMMAND ----------

# MAGIC %md
# MAGIC The loss for our perfect, worst and realistic scenarios was \\(0.002\\), \\(1.502\\) and \\(0.220\\), respectively
# MAGIC 
# MAGIC _NB: in reality, this metric would probably need more consideration and there are certainly edge cases or parameter impact that has not been taken into consideration. Regardless, we will proceed with this for the purpose of this toy exercise._

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enabling Splink
# MAGIC 
# MAGIC As before, let's enable Splink for our notebook

# COMMAND ----------

enable_splink(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting the data (again)
# MAGIC 
# MAGIC And retrieve the data we should already be familiar with.

# COMMAND ----------

# data = spark.read.table(f"splink_{db_name}.companies_with_officers")
table_name = "companies_with_officers"
data = spark.read.format("delta").load(f"dbfs:/Filestore/Users/{username}/{table_name}")

# train_data, test_data = data.randomSplit([0.8, 0.2], seed=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a baseline model
# MAGIC 
# MAGIC Lastly, we'll build a baseline model (using essentially the set-up of the previous demonstration), and log it as the first run in an MLFlow experiment.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We'll set an experiment based on the user ID generated at the start. You can navigate to it via
# MAGIC 
# MAGIC **Left sidebar --> Workspace --> Shared --> \<your-user-name\>_databricks_experiment**

# COMMAND ----------

mlflow.set_experiment(f"/Shared/{db_name}_experiment")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameters
# MAGIC 
# MAGIC Let's set the parameters as in the demonstration notebook.

# COMMAND ----------

blocking_rules_to_generate_predictions = [
  "l.name = r.name and l.date_of_birth = r.date_of_birth",
  "l.nationality = r.nationality and l.locality = r.locality and l.premises = r.premises and l.region = r.region",
  "l.address_line_1 = r.address_line_1 and l.postal_code = r.postal_code and l.surname = r.surname",
]
comparisons = [
  cl.levenshtein_at_thresholds("surname", 10),
  cl.levenshtein_at_thresholds("forenames", 10),
  cl.levenshtein_at_thresholds("address_line_1", 5),
  cl.levenshtein_at_thresholds("country_of_residence", 1),
  cl.levenshtein_at_thresholds("nationality", 2)
]
deterministic_rules = [
  "l.name = r.name and levenshtein(r.date_of_birth, l.date_of_birth) <= 1",
  "l.address_line_1 = r.address_line_1 and levenshtein(l.name, r.name) <= 5",
  "l.name = r.name and levenshtein(l.address_line_1, r.address_line_1) <= 5",
]
settings = {
  "retain_intermediate_calculation_columns": True,
  "retain_matching_columns": True,
  "link_type": "dedupe_only",
  "unique_id_column_name": "uid",
  "em_convergence": 0.01,
  "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions
}


# COMMAND ----------

# MAGIC %md
# MAGIC ### Runnig a trial
# MAGIC 
# MAGIC Let's do an initial run with the above parameters. Notice that this time we're making predictions inside the run as well to also log our loss and the plot generated inside the run.
# MAGIC 
# MAGIC The run should appear in your experiment (you may need to refresh if you have the tab open), or you can navigate to it via the hyperlink provided after the cell has started running.

# COMMAND ----------

with mlflow.start_run() as run:
    RUN_ID = run.info.run_id
  
    model = SplinkLinkerModel()
    model.spark_linker(data)
    model.set_settings(settings)
    model.set_blocking_rules(blocking_rules_to_generate_predictions)
    model.set_comparisons(comparisons)
    model.set_target_rows(1e7)
    model.set_stage1_columns(["name", "date_of_birth"])
    model.set_stage2_columns(["surname", "forenames"])
    
    model.fit(data)
    
    predictions = model.get_linker().predict().as_pandas_dataframe()
    
    model.log_settings_as_json("linker_settings.json")
    
    params = get_hyperparameters(model.get_settings_object())
    all_comparisons = get_all_comparisons(model.get_settings_object())
    charts = get_linker_charts(model.get_linker(), True, True)
    
    mlflow.log_params(params)
    for _ in all_comparisons:
        mlflow.log_params(_)
    mlflow.log_dict(model.get_settings_object(), "linker.json")
    mlflow.pyfunc.log_model("linker", python_model=model)
    
    for name, chart in charts.items():
        model._log_chart(name, chart)
    
    loss, fig = get_match_probabilty_loss(predictions)
    mlflow.log_metric("match_probability_loss", loss)
    mlflow.log_figure(fig, 'prediction_distribution_loss.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieving the baseline model for predictions
# MAGIC 
# MAGIC As before, we can retrieve the model from the experiment (or call it directly from the `model` object created), to make predictions and inspect the results. We stored the ID of the run in `RUN_ID` to make this easier.

# COMMAND ----------

baseline_model = mlflow.pyfunc.load_model(f"runs:/{RUN_ID}/linker")

# COMMAND ----------

baseline_results = baseline_model.predict(data).as_pandas_dataframe()

# COMMAND ----------

baseline_results


# COMMAND ----------

baseline_loss, baseline_fig = get_match_probabilty_loss(baseline_results)

baseline_loss

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn
# MAGIC 
# MAGIC You now have a baseline model which no doubt could use some improvement, which will be up to you to make.
# MAGIC 
# MAGIC Suggestions for work:
# MAGIC 1. Explore the data with Spark/Pandas to better understand the contents of the columns - use Splink's built in profilers to help you.
# MAGIC 2. Use your intuition, your understanding of the data and how the distance functions work to devise a new set or sets of parameters that make more sense given the task.
# MAGIC 3. Train new models and use Splink's built-in visualisers to explore how your predictions were made.
# MAGIC 4. Calculate the loss and compare it against the baseline or previous runs.
# MAGIC 5. Submit new runs to your experiment. Track how the loss evolves with your different trials on the experiments page.
# MAGIC 6. **If you need any help at all, reach out to a TA!**

# COMMAND ----------


