"""
This file contains boilerplate data setup used for our current set of models
"""
import pandas as pd
from ast import literal_eval
from calculate_spectra_stats import simulate_calibration_mistake
from data_generation import generate_calibrated_data


def read_spectra_dataframe(path):
    """
    Reads in a spectrum dataframe and converts its columns which could
    have been saved as strings instead of lists to lists.

    returns dataframe
    """
    df = pd.read_csv(path)
    df['channels'] = df['channels'].apply(literal_eval)
    df['intensities'] = df['intensities'].apply(literal_eval)
    df['masses'] = df['masses'].apply(literal_eval)
    df['masses'] = df['masses'].apply(list)
    return df


def remove_outliers(df, limiter_bool=None):
    """
    Calculates descriptive dataframe stats and then applies a boolean to
    the dataframe to remove outlying spectra. Default boolean aims to remove
    spectra with few peaks, very high masses, or very odd specbinsizes.

    Returns new dataframe with outliers removed / boolean applied.

    Arguments -------
    df: full dataframe of spectra
    limiter_bool: (Optional) boolean expression for indexing the dataframe, e.g
    ((df['SpecBinSize'] < 1) & (df['Number of Peaks'] > 40))
    """
    df['Number of Peaks'] = df['masses'].apply(len)
    df['Minimum Peak Mass'] = df['masses'].apply(min_max_mean_wrapper,
                                                 args=(min, ))
    df['Maximum Peak Mass'] = df['masses'].apply(min_max_mean_wrapper,
                                                 args=(max, ))

    if limiter_bool:
        middle = df[limiter_bool].copy()
    else:
        middle = df[(df['Number of Peaks'] >= 40) &
                    (df['Maximum Peak Mass'] < 20000) &
                    (df['SpecBinSize'] <= 2)].copy()
    middle.reset_index(inplace=True, drop=True)
    return middle


def get_training_data(good_spectra, grey_area, bad_spectra, target_col='Target',
                      num=875, seed=None):
    """
    Takes sets of good, grey, and bad spectra and creates a dataset
    by taking a combination of good, bad, and mis-calibrated good and grey
    spectra.

    Arguments:
    good_spectra: df of good spectra
    grey_area: df of grey spectra
    bad_spectra: df of bad spectra
    target_col: name of target column, default is Target
    num: number of good spectra to sample for mis-calibration
    seed: random seed / state for df.sample
    """
    placeholder = grey_area.reset_index(drop=True)
    miscalled_grey_spectra = miscalibrate_dataframe(placeholder)
    selected = good_spectra.sample(num,
                                   random_state=seed).reset_index(drop=True)
    miscalled_good_spectra = miscalibrate_dataframe(selected)
    miscalibrated = pd.concat([miscalled_grey_spectra,
                               miscalled_good_spectra, bad_spectra])

    to_remove = good_spectra['file_name'].apply(name_in_selected,
                                                args=(selected['file_name'],))
    calibrated = good_spectra[to_remove == False].copy()

    calibrated[target_col] = 1
    miscalibrated[target_col] = 0
    data = pd.concat([calibrated, miscalibrated]).reset_index(drop=True)

    return data


def min_max_mean_wrapper(series, func):
    if len(series) > 0:
        return func(series)
    else:
        return 0


def miscalibrate_dataframe(df):
    """
    Applies simulate calibration mistake to every row of a dataframe of spectra
    data.

    Returns new mis-calibrated dataframe

    Arguments -------
    df: dataframe of spectra data.
    """
    new_df = pd.DataFrame(columns=df.columns)
    for i, row in enumerate(df.itertuples()):
        loc = df.loc[i].copy()
        slope, offset = simulate_calibration_mistake(row.masses,row.channels,
                                                     row.MassOverTime,
                                                     row.MassOffset,
                                                     row.SpecBinSize,
                                                     row.StartFlightTime)
        loc['MassOverTime'] = slope
        loc['MassOffset'] = offset
        new_df.loc[i] = loc
    return generate_calibrated_data(new_df)


def name_in_selected(name, selected_names):
    for string in selected_names:
        if string == name:
            return True
    return False