# Databricks notebook source
# MAGIC %pip install --quiet pytest splink mlflow hyperopt

# COMMAND ----------

dbutils.widgets.text("git_commit", "unknown")

# COMMAND ----------

import pytest
import os
import sys
import datetime

# COMMAND ----------

class ResultsCollector:
    def __init__(self):
        self.reports = []

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if report.when == 'call':
            self.reports.append(report)

# COMMAND ----------

TIME_NOW = datetime.datetime.now()
GIT_COMMIT = dbutils.widgets.get("git_commit")

# COMMAND ----------

user = spark.sql("select current_user()").collect()[0]["current_user()"]

# COMMAND ----------

# Prepare to run pytest from the repo.
os.chdir(f"/Workspace/Repos/{user}/splink-public-sector-hackathon/arc/tests")

sys.dont_write_bytecode = True



# COMMAND ----------

collector = ResultsCollector()
pytest.main(args=[".", "-p","no:cacheprovider"], plugins=[collector])

# COMMAND ----------

test_report = spark.createDataFrame([
  (TIME_NOW, GIT_COMMIT, report.head_line, report.duration, report.outcome, report.passed, report.failed) for report in collector.reports
], ["test_run_timestamp", "git_commit", "test_name", "test_duration", "outcome", "passed", "failed"])

# COMMAND ----------

test_report.display()

# COMMAND ----------


