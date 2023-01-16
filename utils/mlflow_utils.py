import shutil
import mlflow
import os
import json 

from splink.charts import save_offline_chart

# ======================================================================
# ======================================================================
def get_comparison_details(comparison):
    """
    Traverse a single comparison item from the splink settings JSON and log the sql condition the comparison was made
    under and the m and u values
    Parameters
    ----------
    comparison :
    Returns
        A dict containing m and u values and conditions used.
    -------
    """
    output_column_name = comparison['output_column_name']

    comparison_levels = comparison['comparison_levels']
    log_dict = {}
    for _ in comparison_levels:
        sql_condition = _.get('sql_condition')
        m_probability = _.get('m_probability')
        u_probability = _.get('u_probability')
        log_dict[f"output column {output_column_name} compared through condition {sql_condition} m_probability"] = m_probability
        log_dict[f"output column {output_column_name} compared through condition {sql_condition} u_probability"] = u_probability

    return log_dict


# ======================================================================
# ======================================================================
def get_all_comparisons(splink_model_json):
    """
    Traverse the comparisons part of the splink settings and extract the learned values.
    This allows you to easily compare values across different training conditions.
    Parameters
    ----------
    splink_model_json : the settings json from a splink model.
    Returns
        A list of dicts containing m and u values and conditions used.
    -------
    """
    comparisons = splink_model_json['comparisons']
    return [get_comparison_details(cmp) for cmp in comparisons]


# ======================================================================
# ======================================================================
def get_hyperparameters(splink_model_json):
    """
    Simple method for extracting parameters from the splink settings and logging as parameters in MLFlow
    Parameters
    ----------
    splink_model_json : the settings json from a splink model.
    Returns
        The subset of splink_model_json that contains relevant hyperparameters.
    -------
    """
    hyper_param_keys = ['link_type', 'probability_two_random_records_match', 'sql_dialect', 'unique_id_column_name',
                        'em_convergence', 'retain_matching_columns', 'blocking_rules_to_generate_predictions']
    result = {}
    for key in hyper_param_keys:
        result[key] = splink_model_json[key]

    return result


# ======================================================================
# ======================================================================
def get_linker_charts(linker, log_parameters_charts, log_profiling_charts):
    '''
    Log all the non-data related charts to MLFlow
    Parameters
    ----------
    linker : a Splink linker object
    log_parameters_charts: boolean, whether to log parameter charts or not
    log_profiling_charts: boolean, whether to log data profiling charts or not
    Returns
        A dict of charts to be logged for splink model.
    -------
    '''

    charts_dict = {}
    if log_parameters_charts:
        weights_chart = linker.match_weights_chart()
        mu_chart = linker.m_u_parameters_chart()
        compare_chart = linker.parameter_estimate_comparisons_chart()
        charts_dict["weights_chart"] = weights_chart
        charts_dict["mu_chart"] = mu_chart
        charts_dict["compare_chart"] = compare_chart

    if log_profiling_charts:
        missingness_chart = linker.missingness_chart()
        unlinkables = linker.unlinkables_chart()
        blocking_rules_chart = linker.cumulative_num_comparisons_from_blocking_rules_chart()
        charts_dict["missingness_chart"] = missingness_chart
        charts_dict["unlinkables"] = unlinkables
        charts_dict["blocking_rules_chart"] = blocking_rules_chart

    return charts_dict
