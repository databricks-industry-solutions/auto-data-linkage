from typing import Union

from pyspark import SparkContext
from pyspark.sql import Column
from pyspark.sql.column import _to_java_column

ColumnOrName = Union[Column, str]

__all__ = [
    "arc_combinatorial_count_agg",
    "arc_merge_count_map_agg",
    "arc_entropy_agg",
]


# noinspection PyUnresolvedReferences
def _invoke_function(name, *args):
    sc = SparkContext._active_spark_context
    functions_package = getattr(sc._jvm.com.databricks.industry.solutions.arc.functions, "package$")
    functions_object = getattr(functions_package, "MODULE$")
    jf = getattr(functions_object, name)
    return Column(jf(*args))


def arc_combinatorial_count_agg(*attributes: str) -> Column:
    """
    Call arc_combinatorial_count_agg function on the provided attributes.

    Parameters
    ----------
    attributes : *str
        Variable number of attributes to be used in the function.

    Returns
    -------
    Column (MapType(StringType, LongType))

    """
    return _invoke_function(
        "arc_combinatorial_count_agg", *[_to_java_column(c) for c in attributes]
    )


def arc_merge_count_map_agg(map_col: ColumnOrName) -> Column:
    """
    Merge rows of map of counts into a single map.

    Parameters
    ----------
    map_col : ColumnOrName
        Column containing the map of counts.

    Returns
    -------
    Column (MapType(StringType, LongType))

    """
    return _invoke_function(
        "arc_merge_count_map_agg", _to_java_column(map_col)
    )


def arc_entropy_agg(*attributes: str) -> Column:
    """
    Compute entropy of the provided attributes.

    Parameters
    ----------
    attributes : *str
        Variable number of attributes to be used in the function.

    Returns
    -------
    Column (MapType(StringType, DoubleType))

    """
    return _invoke_function(
        "arc_entropy_agg", *[_to_java_column(c) for c in attributes]
    )
