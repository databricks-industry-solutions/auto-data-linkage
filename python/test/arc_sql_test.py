from test.utils import ArcTestCase


class ARC_SQL_Test(ArcTestCase):

    def test_sql(self):
        from arc.sql import functions as f
        df = self.mock_data()

        df.select(
            f.arc_combinatorial_count_agg(2, "col1", "col2", "col3", "col4").alias("count_map")
        ).show()

        df.select(
            f.arc_entropy_agg("col1", "col2", "col3", "col4")
        ).show()
