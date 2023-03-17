import unittest
from importlib.metadata import version

from pyspark.sql import SparkSession


class SparkTestCase(unittest.TestCase):

    spark = None
    library_location = None

    @classmethod
    def setUpClass(cls) -> None:
        SparkSession._instantiatedContext = None
        cls.library_location = f"{arc.__path__[0]}/lib/arc-{version('databricks-arc')}-jar-with-dependencies.jar"
        cls.spark = (
            SparkSession.builder.master("local")
            .config("spark.jars", cls.library_location)
            .getOrCreate()
        )
        cls.spark.conf.set("spark.databricks.industry.solutions.arc.jar.autoattach", "false")
        cls.spark.sparkContext.setLogLevel("warn")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.sparkContext.setLogLevel("warn")
        cls.spark.stop()
