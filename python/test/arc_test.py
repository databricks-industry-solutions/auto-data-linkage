import unittest
from importlib.metadata import version

import arc


class ArcTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # setup testing if needed
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        # teardown testing if needed
        super().tearDownClass()