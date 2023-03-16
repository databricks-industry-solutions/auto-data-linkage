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
def get_function(name):
    sc = SparkContext.getOrCreate()
    functions_object = getattr(sc._jvm.com.databricks.industry.solutions.arc, "functions")
    jf = getattr(functions_object, name)
    return jf


def arc_combinatorial_count_agg(nc, *attributes: str) -> Column:
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
    jf = get_function("arc_combinatorial_count_agg")
    return Column(jf(nc, attributes))


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
    jf = get_function("arc_merge_count_map_agg")
    return Column(jf(_to_java_column(map_col)))


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
    jf = get_function("arc_entropy_agg")
    return Column(jf(attributes))
