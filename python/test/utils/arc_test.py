import unittest
from importlib.metadata import version

from .spark_test_case import SparkTestCase

class ArcTestCase(SparkTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # setup testing if needed
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        # teardown testing if needed
        super().tearDownClass()

    def mock_data(self):
        return self.spark.createDataFrame(
            [
                (1, 1, 1, 1, 1),
                (2, 2, 2, 2, 1),
                (3, 3, 3, 2, 1),
                (4, 3, 4, 3, 1),
                (5, 3, 5, 3, 1),
                (3, 3, 3, 3, 1),
                (4, 4, 5, 4, 1),
                (4, 4, 4, 4, 1),
                (5, 5, 5, 5, 1),
            ],
            ["id", "col1", "col2", "col3", "col4"],
        )