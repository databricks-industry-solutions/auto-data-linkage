from test.utils import ArcTestCase


class TestARC_SQL(ArcTestCase):

    def test_sql(self):
        from arc.sql import functions as f
        df = self.mock_data()

        df.select(
            f.arc_combinatorial_count_agg(2, "col1", "col2", "col3", "col4").alias("count_map")
        ).show()

        df.select(
            f.arc_entropy_agg(0, "col1", "col2", "col3", "col4"),
            f.arc_entropy_agg(10, "col1", "col2", "col3", "col4")
        ).show()


    def test_blocking_rules(self):
        from arc.sql import functions as f
        df = self.mock_data()

        f.arc_generate_blocking_rules(df, 2, 2, "col1", "col2", "col3", "col4").show()
