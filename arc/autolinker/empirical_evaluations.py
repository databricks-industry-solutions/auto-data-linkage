def calculate_empirical_score(data, predictions, unique_id, threshold):
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
  df_true_positives = df_true

  # convert predictions to Spark DataFrame and filter on match prob threshold - table will only contain predicted positives
  df_pred = predictions.as_spark_dataframe().filter(f"match_probability>={threshold}")

  # Calculate TP, FP, FN, TN

  # TP is the inner join of true and predicted pairs
  tp = df_pred.join(
    df_true_positives,
    ((df_pred[f"{unique_id}_l"]==df_true_positives[f"{unique_id}_r"]) & (df_pred[f"{unique_id}_l"]==df_true_positives[f"{unique_id}_r"])) | ((df_pred[f"{unique_id}_l"]==df_true_positives[f"{unique_id}_r"]) & (df_pred[f"{unique_id}_r"]==df_true_positives[f"{unique_id}_l"])),
    how="inner"
  ).count()

  # FN is the left anti-join of true and predicted
  fn = df_true_positives.join(
    df_pred,
    ((df_pred[f"{unique_id}_l"]==df_true_positives[f"{unique_id}_l"]) & (df_pred[f"{unique_id}_r"]==df_true_positives[f"{unique_id}_r"])) | ((df_pred[f"{unique_id}_l"]==df_true_positives[f"{unique_id}_r"]) & (df_pred[f"{unique_id}_r"]==df_true_positives[f"{unique_id}_l"])),
    how="left_anti"
  ).count()

  # FP is the left anti-join of predicted and true
  fp = df_pred.join(
    df_true_positives,
    ((df_pred[f"{unique_id}_l"]==df_true_positives[f"{unique_id}_l"]) & (df_pred[f"{unique_id}_r"]==df_true_positives[f"{unique_id}_r"])) | ((df_pred[f"{unique_id}_l"]==df_true_positives[f"{unique_id}_r"]) & (df_pred[f"{unique_id}_r"]==df_true_positives[f"{unique_id}_l"])),
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