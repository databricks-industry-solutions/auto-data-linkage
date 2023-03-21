import importlib.metadata
import importlib.resources

from .functions import LibraryHandler
from pyspark import SparkContext
from pyspark.sql import SparkSession


def enable_arc():
    spark = SparkSession.getActiveSession()
    if spark is not None:
        _auto_attach_enabled = (
            spark.conf.get("spark.databricks.industry.solutions.jar.autoattach", "false") == "true"
        )

        if _auto_attach_enabled:
            _ = LibraryHandler(spark)
