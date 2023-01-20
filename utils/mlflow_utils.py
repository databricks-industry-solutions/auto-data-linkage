import shutil
import mlflow
import os
import json 
import math
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

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


# ======================================================================
# ======================================================================
def get_match_probabilty_loss(predictions):
    '''
    Calculates the custom loss of a pariwise Splink prediction dataframe, by
    1. Fitting 2 Gaussians on the probability distribution
    2. Calculating and summing the means from 0 and 1 (respectively) and the standard deviations of the Gaussians
    Parameters
    ----------
    predictions: pandas.DataFrame, df of predictions that must contain the match_probabilty column
    Returns
        A 2-tuple of the loss (float) and a chart (matplotlib.pyplot.figure) of the distribution and the fit
    '''
    
    # check if match_probability column is in the dataframe
    if "match_probability" in predictions.columns:
        pred = predictions["match_probability"].values
    else:
        raise Exception("The match_probability column wasn't found in dataframe. You need to pass a dataframe that was predicted by Splink.")
        
    # Fig Gaussian mixture model (disable autologging for this)
    mlflow.autolog(disable=True)
    gm = GaussianMixture(n_components=2, random_state=0).fit(pred.reshape(-1, 1))
    mlflow.autolog(disable=False)
    
    # Get means, variances and standard deviations
    mean_1 = gm.means_[0][0]
    mean_2 = gm.means_[1][0]
    cov_1 = gm.covariances_[0][0][0]
    cov_2 = gm.covariances_[1][0][0]
    std_1 = math.sqrt(cov_1)
    std_2 = math.sqrt(cov_2)

    # Calculate loss
    mean_goodness = mean_1 + 1.0 - mean_2
    std_goodness = std_1 + std_2
    prediction_purity = mean_goodness + std_goodness

    # Create probability density functions based on the two sets of parameters
    x = np.linspace(0.0, 1.0, 100)
    gauss_1 = ss.norm.pdf(x, mean_1, std_1)
    gauss_2 = ss.norm.pdf(x, mean_2, std_2)

    # plot chart
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(pred, density=True, bins=50)
    ax.plot(x, gauss_1, c="green", linewidth=3)
    ax.plot(x, gauss_2, c="orange", linewidth=3)
    ax.axvline(mean_1, c="green", linewidth=1, linestyle="--")
    ax.axvline(mean_2, c="orange", linewidth=1, linestyle="--")
    ax.axvline(0.0, c="grey", linewidth=1, linestyle="--")
    ax.axvline(1.0, c="grey", linewidth=1, linestyle="--")
    ax.hlines(40, 0.0, mean_1, linewidth=2, linestyle="-", color="red")
    ax.hlines(40, mean_2, 1.0, linewidth=2, linestyle="-", color="red")
    ax.hlines(40, mean_1, mean_1+cov_1, linewidth=2, linestyle="-", color="yellow")
    ax.hlines(45, mean_2-cov_2, mean_2, linewidth=2, linestyle="-", color="yellow")
    ax.set_title("Probability density of predictions and fitted parameters")
    
    return prediction_purity, fig
