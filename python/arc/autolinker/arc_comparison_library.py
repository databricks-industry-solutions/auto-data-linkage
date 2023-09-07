from typing import Dict

def generate_custom_rule(
        column_name: str,
        threshold: float,
        sql_function_name: str
    ) -> Dict:
    """
    Function to generate raw dictionary describing a Spark Splink compatible
    comparison.

    Returns a dictionary of comparison levels.

    :param column_name: string name of column to be compared
    :param threshold: float threshold of the first comparison level
    :param sql_function_name: name of the registered Spark SQL UDF
    """

    comparison = {
        "output_column_name": column_name,
        "comparison_description": f"{column_name} cosine distance",
        "comparison_levels": [
            {
                "sql_condition": f"{column_name}_l IS NULL OR {column_name}_r IS NULL",
                "label_for_charts": "Null",
                "is_null_level": True,
            },
            {
                "sql_condition": f"{column_name}_l = {column_name}_r",
                "label_for_charts": "Exact match"
            },
            {
                "sql_condition": f"{sql_function_name}({column_name}_l, {column_name}_r) &gt; {str(threshold)}",
                "label_for_charts": f"{sql_function_name} > {str(threshold)}"
            },
            {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
        ],
    }

    return comparison


