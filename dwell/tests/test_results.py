import pickle
import numpy as np
import pandas as pd

def load_pickle(filename):
    """ Load a pickle file. """
    with open(filename, "rb") as f:
        return pickle.load(f)

def compare_dataframes(generated_df, expected_df, filename, atol=1e-4, rtol=1e-3):
    """ Compare two DataFrames and print differences if they are not equal. """
    try:
        pd.testing.assert_frame_equal(generated_df, expected_df, check_dtype=False,  # Ignore dtype mismatches
                                      atol=atol,  # Absolute tolerance
                                      rtol=rtol)  # Relative tolerance
    except AssertionError as e:
        # Print mismatched values
        diff = generated_df - expected_df  # Element-wise difference
        max_diff = np.abs(diff).max().max()  # Maximum absolute difference
        mean_diff = np.abs(diff).mean().mean()  # Mean absolute difference

        print(f"Mismatch detected in {filename}")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print("\nFirst few mismatched rows:\n")

        # Identify mismatched rows
        mismatched_rows = diff[diff.abs() > atol].dropna()
        print(mismatched_rows.head(10))  # Print first 10 mismatched rows

        raise AssertionError(f"Mismatch in {filename}:\n{e}")

def test_single_phase_non_isothermal_outputs():
    """ Test results from examples/single_phase_non_isothermal/main_validation_single_phase_of_CO2.py. """
    generated_file = "examples/single_phase_non_isothermal/stored_primary_variables.pkl"
    expected_file = "tests/expected_results/main_validation_single_phase_of_CO2/expected_stored_primary_variables.pkl"

    generated_df = load_pickle(generated_file)
    expected_df = load_pickle(expected_file)
    compare_dataframes(generated_df, expected_df, generated_file)

def test_two_phase_isothermal_outputs():
    """ Test results from examples/two_phase/main_full_column_of_water_variable_K_2-comp_validation_scenario2.py. """
    generated_file = "examples/two_phase/stored_primary_variables.pkl"
    expected_file = "tests/expected_results/main_full_column_of_water_variable_K_2-comp_validation_scenario2/expected_stored_primary_variables.pkl"

    generated_df = load_pickle(generated_file)
    expected_df = load_pickle(expected_file)
    compare_dataframes(generated_df, expected_df, generated_file)
