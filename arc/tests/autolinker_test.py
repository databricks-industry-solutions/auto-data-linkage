import pytest
from arc.autolinker.autolinker import AutoLinker
from pyspark.sql import SparkSession
import splink.spark.spark_comparison_library as cl
import hyperopt.pyll.stochastic



def test_calculate_column_entropy():
  spark = SparkSession.builder.appName('abc').getOrCreate()
  # mock data
  df_mock = spark.createDataFrame([
    (0, "a"),
    (1, "a"),
    (2, "b"),
    (3, "b"),
    (4, "b"),
    (5, "c"),
    (6, "c"),
    (7, "c"),
    (8, "c"),
    (9, "c"),
  ], ["id", "dummy"])

  autolinker = AutoLinker()

  # set row count manually
  autolinker.data_rowcount = df_mock.count()

  actual_entropy = autolinker._calculate_column_entropy(df_mock, "dummy")

  expected_entropy = 1.0296530140645737

  assert actual_entropy==expected_entropy


def test_generate_rules():
  mock_columns = [
    ["a"],
    ["a", "b"],
    ["a", "b", "c"],
    ["a", "a"]
  ]

  autolinker = AutoLinker()

  expected_output = [
    ['l.a = r.a'],
    ['l.a = r.a', 'l.b = r.b'],
    ['l.a = r.a', 'l.b = r.b', 'l.c = r.c'],
    ['l.a = r.a', 'l.a = r.a']
  ]

  actual_output = [autolinker._generate_rules(c) for c in mock_columns]

  assert expected_output==actual_output


def test_generate_candidate_blocking_rules():
  spark = SparkSession.builder.appName('abc').getOrCreate()
  # mock data
  df_mock = spark.createDataFrame([
    (0, "a"),
    (1, "a"),
    (2, "b"),
    (3, "b"),
    (4, "b"),
    (5, "c"),
    (6, "c"),
    (7, "c"),
    (8, "c"),
    (9, "c"),
  ], ["id", "dummy"])

  autolinker = AutoLinker()

  expected_output = [['l.dummy = r.dummy']]

  actual_output = autolinker._generate_candidate_blocking_rules(df_mock, ["dummy"], 100, "id", max_columns_per_rule=2)

  assert expected_output==actual_output


def test_clean_columns():
  spark = SparkSession.builder.appName('abc').getOrCreate()
  # mock data
  df_mock = spark.createDataFrame([
    (0, "aaaa"),
    (1, "a123"),
    (2, "aaa aaa"),
    (3, " a 1 a 1"),
    (4, "A 1 A 999"),
    (5, "a---[a][a][]"),
    (6, " [ ] [ ]"),
    (7, "___#aasdf"),
    (8, "___")
  ], ["id", "dummy"])

  autolinker = AutoLinker()

  expected_output = spark.createDataFrame([
    (0, "aaaa"),
    (1, "a123"),
    (2, "aaa aaa"),
    (3, " a 1 a 1"),
    (4, "a 1 a 999"),
    (5, "aaa"),
    (6, "    "),
    (7, "aasdf"),
    (8, "")
  ], ["id", "dummy"])

  actual_output = autolinker._clean_columns(df_mock, ["dummy"], cleaning="all")

  assert expected_output.schema==actual_output.schema and expected_output.collect()==actual_output.collect()


def test_create_comparison_list():
  mock_space = {"comparisons": {
    "a": {
      "distance_function": {
        "distance_function": "jaccard",
        "threshold": 0.5
      }
    },
    "b": {
      "distance_function": {
        "distance_function": "levenshtein",
        "threshold": 1
      }
    },
    "c": {
      "distance_function": {
        "distance_function": "jaro_winkler",
        "threshold": 0.5
      }
    }
  },
  "blocking_rules": ["l.a = r.a"]
  }

  autolinker = AutoLinker()

  expected_output = [
    cl.jaccard_at_thresholds("a", 0.5),
    cl.levenshtein_at_thresholds("b", 1),
    cl.jaro_winkler_at_thresholds("c", 0.5)
  ]

  actual_output = autolinker._create_comparison_list(mock_space)


  assert all([expected._comparison_dict["comparison_description"]==actual._comparison_dict["comparison_description"] for expected, actual in zip(expected_output, actual_output)])


def test_calculate_entropy_delta():
  spark = SparkSession.builder.appName('abc').getOrCreate()
  # mock data
  df_mock = spark.createDataFrame([
    (0, "a"),
    (1, "a"),
    (2, "b"),
    (3, "b"),
    (4, "b"),
    (5, "c"),
    (6, "c"),
    (7, "c"),
    (8, "c"),
    (9, "c"),
  ], ["id", "dummy"])
  
  df_mock_dedupe = spark.createDataFrame([
    (0, "a"),
    (3, "b"),
    (4, "b"),
    (8, "c"),
    (9, "c"),
  ], ["id", "dummy"])

  expected_output = -0.15561933979152887

  autolinker = AutoLinker()

  # set row count manually
  autolinker.data_rowcount = df_mock.count()
  actual_output = autolinker.calculate_entropy_delta(df_mock, df_mock_dedupe, ["dummy"])

  assert expected_output==actual_output


def test_create_hyperopt_space():
  spark = SparkSession.builder.appName('abc').getOrCreate()
  # mock data
  df_mock = spark.createDataFrame([
    (0, "a"),
    (1, "a"),
    (2, "b"),
    (3, "b"),
    (4, "b"),
    (5, "c"),
    (6, "c"),
    (7, "c"),
    (8, "c"),
    (9, "c"),
  ], ["id", "dummy"])

  autolinker = AutoLinker()

  space = autolinker._create_hyperopt_space(df_mock, ["dummy"], 1000, "id", max_columns_per_rule=2)

  sample_space = hyperopt.pyll.stochastic.sample(space)

  # keys exist
  keys = ["blocking_rules", "comparisons"]
  keys_exist = all([key in sample_space.keys() for key in keys])

  #blocking rule type
  blocking_rule_type = type(sample_space["blocking_rules"]) == tuple

  # correct function
  distance_functions = ["jaccard", "levenshtein", "jaro_winkler"]
  correct_function = any([sample_space["comparisons"]["dummy"]["distance_function"]["distance_function"] in df for df in distance_functions])

  # correct threshold
  correct_threshold = sample_space["comparisons"]["dummy"]["distance_function"]["threshold"]>0.0

  assert all([keys_exist, blocking_rule_type, correct_function, correct_threshold])


def test_drop_intermediate_tables():
  spark = SparkSession.builder.appName('abc').getOrCreate()

  # create dummy tables in test catalog.schema
  spark.sql("create or replace table marcell_splink.test_tables.test__splink__abc as select 'a' as test_col") # table to delete
  spark.sql("create or replace table marcell_splink.test_tables.__splink__abc as select 'a' as test_col") # table to delete
  spark.sql("create or replace table marcell_splink.test_tables.test__splink__ as select 'a' as test_col") # table to delete
  spark.sql("create or replace table marcell_splink.test_tables.test_dont_delete as select 'a' as test_col") # table to ignore

  tables_before = spark.sql("show tables from marcell_splink.test_tables").collect()

  # check if __splink__ tables exist in the database before
  splink_exists_before = any(["__splink__" in table.tableName for table in tables_before])

  autolinker = AutoLinker(
    catalog="marcell_splink",
    schema="test_tables"
  )
  autolinker.spark = spark

  # drop tables
  autolinker._drop_intermediate_tables()

  # collect tables again
  tables_after = spark.sql("show tables from marcell_splink.test_tables").collect()

  # check that __splink__ tables have been deleted
  splink_tables_deleted = all(["__splink__" not in table.tableName for table in tables_after])

  # check if the dummy table is still there
  dummy_table_exists = any(["test_dont_delete" == table.tableName for table in tables_after])

  assert all([splink_exists_before, splink_tables_deleted, dummy_table_exists])


def test_convert_hyperopt_to_splink():
  spark = SparkSession.builder.appName('abc').getOrCreate()
  df_mock = spark.createDataFrame([
    (0, "a", "1"),
    (1, "a", "1"),
    (2, "b", "2"),
    (3, "b", "2"),
    (4, "b", "3"),
    (5, "c", "6"),
    (6, "c", "7"),
    (7, "c", "8"),
    (8, "c", "9"),
    (9, "c", "0"),
  ], ["id", "dummy1", "dummy2"])

  autolinker = AutoLinker(
    catalog="marcell_splink",
    schema="test_tables"
  )
  autolinker.spark = spark

  # train autolinker
  autolinker.auto_link(
    df_mock,
    attribute_columns=["dummy1", "dummy2"],
    unique_id="id",
    comparison_size_limit=1000,
    max_evals=1
  )

  # convert best trial to dictionary
  output = autolinker._convert_hyperopt_to_splink()

  # keys exist
  keys = ["blocking_rules", "comparisons"]
  keys_exist = all([key in output.keys() for key in keys])

  #blocking rule type
  blocking_rule_type = type(output["blocking_rules"]) == list

  # correct functions
  distance_functions = ["jaccard", "levenshtein", "jaro_winkler"]
  correct_function_dummy1 = any([output["comparisons"]["dummy1"]["distance_function"]["distance_function"] in df for df in distance_functions])
  correct_function_dummy2 = any([output["comparisons"]["dummy2"]["distance_function"]["distance_function"] in df for df in distance_functions])
  # correct threshold
  correct_threshold_dummy1 = output["comparisons"]["dummy1"]["distance_function"]["threshold"]>0.0
  correct_threshold_dummy2 = output["comparisons"]["dummy2"]["distance_function"]["threshold"]>0.0

  assert all([keys_exist, blocking_rule_type, correct_function_dummy1, correct_function_dummy2, correct_threshold_dummy1, correct_threshold_dummy2])


def test_randomise_columns():
  autolinker = AutoLinker()

  # test with 2 columns
  test_attribute_columns_2 = ["a", "b"]
  expected_randomised_columns_2 = ["a", "b"]

  actual_randomised_columns_2 = autolinker._randomise_columns(test_attribute_columns_2)

  two_columns_test = expected_randomised_columns_2==actual_randomised_columns_2

  # test with 3 columns
  test_attribute_columns_3 = ["a", "b", "c"]
  expected_randomised_columns_3 = ["a&b", "a&c", "b&c", "b&a", "c&a", "c&b"]

  actual_randomised_columns_3 = autolinker._randomise_columns(test_attribute_columns_3)

  three_columns_test = all([col in expected_randomised_columns_3 for col in actual_randomised_columns_3])


  assert all([two_columns_test, three_columns_test])
