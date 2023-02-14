# Databricks notebook source
pip install git+https://github.com/robertwhiffin/splink.git@mllfow-integration

# COMMAND ----------

import pandas as pd
import numpy as np

from utils.mlflow_utils import get_match_probabilty_loss

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating Splink
# MAGIC 
# MAGIC Whilst Splink does offer a supervised predictive method, our data does not have ground truth (we don't categorically know which records are duplicates or otherwise), so we're resigned to taking an unsupervised approach.
# MAGIC 
# MAGIC Unsupervised models are trickier to evaluate, because we can't compare the predictions against a known truth. How would we know how well our model has performed then? How can we evaluate the impact of changing parameters against the previous experiment?
# MAGIC 
# MAGIC We are proposing a loss function that promotes match probabilities within the pairwise prediction dataframe to be close to either 1 or 0 (i.e. the model is "certain" that a pair is a match or not).
# MAGIC 
# MAGIC ##### A word of caution
# MAGIC 
# MAGIC Please be aware that this is not a hard-and-fast, empirical way of evaluating a Splink model. Unsupervised entity resolution is complex and choosing the right model will depend on the data, use case, domain, expert knowledge, risk appetite, and many more factors.

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
