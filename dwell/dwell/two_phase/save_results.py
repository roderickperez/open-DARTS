import pandas as pd

def save_in_df(initial_primary_variables):
    """
    Initialize an empty DataFrame and save the primary variables (initial conditions) in that DataFrame

    :param initial_primary_variables: Initial primary variables
    :type initial_primary_variables: ndarray or list
    :return: The data frame in which the data are stored so that it can be used later by the function add_to_df.
    """
    # Initialize an empty DataFrame to store the primary variables
    primary_variables_df = pd.DataFrame()

    column_name = 'Initial conditions'

    results_series = pd.Series(initial_primary_variables, name=column_name)

    primary_variables_df[column_name] = results_series

    return primary_variables_df

def add_to_df(primary_variables, column_name, primary_variables_df):
    """
    Add the primary variables of a wellbore do the DataFrame in which some results are already stored by the function save_in_df.

    :param primary_variables: Primary variables
    :type primary_variables: ndarray or list
    :param column_name: Name of the column in which the primary variables will be written, e.g., "Step 1", etc.
    :type column_name: str
    :param primary_variables_df: The data frame returned by the function save_in_df
    :type primary_variables_df: DataFrame
    :return:
    """
    results_series = pd.Series(primary_variables, name=column_name)

    primary_variables_df = pd.concat([primary_variables_df, results_series.to_frame()], axis=1)

    return primary_variables_df
