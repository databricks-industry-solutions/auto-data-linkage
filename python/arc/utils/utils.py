from mlflow import MlflowClient

def delete_all_default_linking_experiments(spark):
    '''
    This method removes all MLFlow experiments created by ARC with the default naming convention.

    :param spark: a spark object. available by default in databricks.


    Returns
    -------

    '''
    if input("WARNING - this will delete all your ARC generated MLFlow experiments, Type 'YES' to proceed") != "YES":
        return
    username = spark.sql('select current_user() as user').collect()[0]['user']
    pattern = f"%{username}/Databricks Autolinker%"

    client = MlflowClient()
    experiments = (
        client.search_experiments(filter_string=f"name LIKE '{pattern}'")
    )  # returns a list of mlflow.entities.Experiment
    for exp in experiments:
        client.delete_experiment(exp.experiment_id)