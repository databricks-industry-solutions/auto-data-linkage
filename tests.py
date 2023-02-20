# Databricks notebook source
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion


# COMMAND ----------


job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_automation",
            "group": "RCG"
        },
        "tasks": [
            {
                "job_cluster_key": "hackathon_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"/Workshop Materials/notebooks/Splink in Databricks - Expert Answer",
                  "base_parameters": {
                    "testing": "True"
                    }
                },
                "task_key": "workshop-expert_answer",
                "description": ""
            },
          {
                "job_cluster_key": "hackathon_cluster",
                "libraries": [],
                "notebook_task": {
                  "notebook_path": f"/Workshop Materials/notebooks/Splink in Databricks - end-to-end example",
                  "base_parameters": {
                    "testing": "True"
                    }
                },
                "task_key": "workshop-end_to-end_example",
                "description": ""
            },
          {
                "job_cluster_key": "hackathon_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"/Workshop Materials/notebooks/Splink in Databricks - Exercise",
                  "base_parameters": {
                    "testing": "True"
                    }
                },
                "task_key": "workshop-exercise",
                "description": ""
            },
            
        ],
        "job_clusters": [
            {
                "job_cluster_key": "hackathon_cluster",
                "new_cluster": {
                    "spark_version": "12.1.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.preview.enabled": "true"
                    },
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_D4ads_v5", "GCP": "n1-highmem-4"}, # different from standard API
                    "custom_tags": {
                        "usage": "solacc_automation"
                    },
                }
            }
        ]
    }



dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)
