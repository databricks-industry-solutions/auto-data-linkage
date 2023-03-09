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
GIT_COMMIT


# COMMAND ----------

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# Get the repo's root directory name.
repo_root = os.path.dirname(os.path.dirname(notebook_path))

# Prepare to run pytest from the repo.
root_dir = f"/Workspace{repo_root}"
print(os.getcwd())
sys.dont_write_bytecode = True


# COMMAND ----------

collector = ResultsCollector()
pytest.main(args=[root_dir, "-p","no:cacheprovider"], plugins=[collector])

# COMMAND ----------

test_report = spark.createDataFrame([
  (TIME_NOW, GIT_COMMIT, report.head_line, report.duration, report.outcome, report.passed, report.failed) for report in collector.reports
], ["test_run_timestamp", "git_commit", "test_name", "test_duration", "outcome", "passed", "failed"])

# COMMAND ----------

test_report.display()

# COMMAND ----------


