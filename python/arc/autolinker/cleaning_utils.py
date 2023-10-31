from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import typing

def _clean_columns(
        cleaning_mode: str,
        data: DataFrame,
        attribute_columns: list):
    """
    Method to clean string columns (turn them to lower case and remove non-alphanumeric characters)
    in order to help with better (and quicker) string-distance calculations. If cleaning is 'all'
    (as is by default), it will automatically clean as much as it can.  If cleaning is 'none', it will do nothing.
    cleaning can also be a dictionary with keys as column names and values as lists of method strings.
    The currently available cleaning methods are turning to lower case and removing non-alphanumeric characters.

    Returns a Spark DataFrame.

    :param data: DataFrame containing the data to be cleaned
    :param attribute_columns: all attribute columns
    :param cleaning: string or dicitonary with keys as column names and values as list of valid cleaning method names.

    """

    cleaning = cleaning_mode
    # if cleaning is secleaningt to "none", don't do anything to the data
    if cleaning_mode == "none":
        return data

    # if it's set to "all", turn it into a dictionary covering all possible cases
    if cleaning == "all":
        cleaning = {col: ["lower", "alphanumeric_only"] for col in attribute_columns}

    for col, methods in cleaning.items():
        # if column is not a string, skip it
        if not data.schema[col].dataType == StringType():
            continue

        for method in methods:
            if method == "alphanumeric_only":
                # replace column and only keep alphanumeric and space characters
                data = data.withColumn(col, F.regexp_replace(F.col(col), r"[^A-Za-z0-9 ]+", ""))

            elif method == "lower":
                data = data.withColumn(col, F.lower(F.col(col)))

    return data


def _do_column_cleaning(
        cleaning_mode
        , autolink_data
        , linker_mode
        , attribute_columns
) -> typing.Union[DataFrame, list]:
    # Clean the data
    if linker_mode == "dedupe_only":
        output_data = _clean_columns(cleaning_mode, autolink_data, attribute_columns)
    else:
        output_data = [
            _clean_columns(cleaning_mode, autolink_data[0], attribute_columns),
            _clean_columns(cleaning_mode, autolink_data[1], attribute_columns)
        ]
    return output_data

