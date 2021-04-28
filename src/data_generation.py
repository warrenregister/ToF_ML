"""
DataFrame Generation and Data Transformation Functions
"""
from os import listdir

import numpy as np
import pandas as pd


def add_error(number, modifier=2, tens=2, threshold=0.5,
              sub_one=False) -> tuple:
    """
    Add or subtract a random amount of error based on the modifier and tens
    variables.
    Error calculated : (random num between 0 and 1 / modifier) / ten**tens
    """
    randoms = np.random.rand(2)
    error = (randoms[0] / modifier) / 10 ** tens
    if sub_one:
        error = (1 - randoms[0] / modifier) / 10 ** tens
    mult = 1 if randoms[1] > threshold else -1
    new_num = number + mult * number * error
    return new_num, error * mult


def get_amu_channel(spec_bin_size, start_time, mass_over_time, mass_offset,
                    amu=100) -> np.array:
    """
    Calculate the channel which corresponds to 100 AMU for a spectra slope
    and offset values.
    """
    return (((amu ** .5 - mass_offset) / mass_over_time) - start_time) * 1 / \
           (.001 * spec_bin_size)


def get_amu_offset(spec_bin_size, start_time, mass_over_time, channel,
                   amu=100) -> np.array:
    """
    Calculate the offset which moves a channel from its current AMU to the
    given AMU
    """
    return amu ** .5 - (channel * (.001 * spec_bin_size) + start_time) *\
        mass_over_time


def get_amu_slope(spec_bin_size, start_time, mass_offset, channel,
                  amu=100) -> np.array:
    """
    Calculate the slope which moves a channel from its current AMU to the
    given AMU
    """
    return (amu ** .5 - mass_offset) / (channel * (.001 * spec_bin_size) +
                                        start_time)


def generate_data(data, tens, modifier, ranges=(0.2, 0.4, .6),
                  multi_class=False, slope_cat=False, sub_one=False,
                  bad_amt=.0035, good_amt=.0015, slope_mods=False,
                  slope_modifiers=(3, 2)) -> pd.DataFrame:
    """
    Takes calibrated ToF-MS data and adds error to the offset and mass.
    Returns classification dataset 0: acceptable error, 1: unacceptable error
    If multi_class:1: offset error, 2: slope error, 3: both. Does not
    recalibrate with new slope and offset values.


    Arguments -------
    data: dataframe to add error to
    tens: power of tens place for error
    modifier: modifier to apply to error, eg (rand num) / modifier
    ranges: percentile ranges for different outcomes slope error, offset
    and slope error, offset error
    multi_class: if True creates mutliclass dataset, else creates binary dataset
    slope_cat: if true both and slope error are in same the category of target.
    sub_one: sub one argument for add error, whether to subtract from 1, flips
    range when modifier is greater than 1.
    slope_mods: boolean to indicate that a different set of modifiers should be
                used to calculate slope
    slope_modifiers: a 2 element list containing [tens, modifier] for slope
    """
    err_data = data.copy()
    target = []
    error_percent_slope = []
    error_percent_offset = []
    err_amt = []
    new_slope = []
    new_offset = []
    for row in err_data.itertuples():
        chan_100 = get_amu_channel(row.SpecBinSize, row.StartFlightTime,
                                   row.MassOverTime, row.MassOffset)
        init_100 = mass_formula(np.array(chan_100), row.SpecBinSize,
                                row.StartFlightTime, row.MassOverTime,
                                row.MassOffset)

        num = np.random.rand(1)
        if num < ranges[0]:  # slope only
            ten = tens
            mod = modifier
            if slope_mods:
                ten = slope_modifiers[0]
                mod = slope_modifiers[1]

            slope, sl_err = add_error(row.MassOverTime, tens=ten, modifier=mod,
                                      sub_one=sub_one)
            error_percent_slope.append(sl_err)
            new_slope.append(slope)
            new_offset.append(row.MassOffset)
            error_percent_offset.append(0)
            new_100 = mass_formula(np.array(chan_100), row.SpecBinSize,
                                   row.StartFlightTime, slope,
                                   row.MassOffset)
            diff_100 = init_100 - new_100
            err_amt.append(diff_100)
            diff_100 = abs(diff_100)
            if abs(diff_100) > good_amt:
                if multi_class:
                    target.append(2)
                else:
                    if diff_100 > bad_amt:
                        target.append(2)
                    else:
                        target.append(1)
            else:
                target.append(0)
        elif ranges[0] <= num < ranges[1]:  # both
            ten = tens
            mod = modifier
            if slope_mods:
                ten = slope_modifiers[0]
            offset, off_err = add_error(row.MassOffset, tens=tens,
                                        modifier=modifier, sub_one=sub_one)
            error_percent_offset.append(off_err)
            new_offset.append(offset)
            slope, sl_err = add_error(row.MassOverTime, tens=ten, modifier=mod,
                                      sub_one=sub_one)
            error_percent_slope.append(sl_err)
            new_slope.append(slope)
            new_100 = mass_formula(np.array(chan_100), row.SpecBinSize,
                                   row.StartFlightTime, slope, offset)
            diff_100 = init_100 - new_100
            err_amt.append(diff_100)
            if abs(diff_100) > good_amt:
                if multi_class:
                    if slope_cat:
                        target.append(2)
                    else:
                        target.append(3)
                else:
                    if diff_100 > bad_amt:
                        target.append(2)
                    else:
                        target.append(1)
            else:
                target.append(0)
        elif ranges[1] <= num < ranges[2]:  # offset only
            error_percent_slope.append(0)
            offset, off_err = add_error(row.MassOffset, tens=tens,
                                        modifier=modifier, sub_one=sub_one)
            error_percent_offset.append(off_err)
            new_offset.append(offset)
            new_slope.append(row.MassOverTime)
            new_100 = mass_formula(np.array(chan_100), row.SpecBinSize,
                                   row.StartFlightTime, row.MassOverTime,
                                   offset)
            diff_100 = init_100 - new_100
            err_amt.append(diff_100)
            if abs(diff_100) > good_amt:
                if diff_100 > bad_amt:
                    target.append(2)
                else:
                    target.append(1)
            else:
                target.append(0)
        else:  # none
            target.append(0)
            error_percent_slope.append(0)
            error_percent_offset.append(0)
            err_amt.append(0)
            new_slope.append(row.MassOverTime)
            new_offset.append(row.MassOffset)

    err_data['target'] = target
    err_data['err_prop_slope'] = error_percent_slope
    err_data['err_prop_offset'] = error_percent_offset
    err_data['err_at_100amu'] = err_amt
    err_data['MassOverTime'] = new_slope
    err_data['MassOffset'] = new_offset
    return err_data


def mass_formula(channels: np.array, spec_bin_size, start_time, mass_over_time,
                 mass_offset) -> np.array:
    """
    Fast conversion from flight time to mass.
    """
    return ((channels * .001 * spec_bin_size + start_time) * mass_over_time +
            mass_offset) ** 2


def generate_calibrated_data(data) -> pd.DataFrame:
    """
    Applies mass_formula to every row in dataset to allow
    calibrated graphs to be generated.
    """
    new_data = data.copy()
    masses = []
    for row in new_data.itertuples():
        spec = row.SpecBinSize
        m_over_t = row.MassOverTime
        m_offset = row.MassOffset
        time = row.StartFlightTime
        masses.append(mass_formula(np.array(row.channels), spec, time,
                                   m_over_t, m_offset))
    new_data['masses'] = masses
    return new_data


def get_precise_peaks(df, names):
    """
    Return list of peaks channels and intensities from dataframe of precise
    peaks.
    """
    all_peaks = []
    for row in df.iterrows():
        chans = row[1][names[0]][row[1][names[1]] > 0]
        intens = row[1][names[1]][row[1][names[1]] > 0]
        all_peaks.append(list(zip(chans, intens)))
    return all_peaks


def get_better_spectra(loc='../data/SpectraCsvFiles_WatsonPeakFinder/')\
        -> pd.DataFrame:
    """
    Read in better peak info and transform into lists of channels, intensities,
    names.
    """
    channels = []
    intensities = []
    names = []
    for csv in listdir(loc):
        data = pd.read_csv(loc + csv, names=['channels', 'intensities'])
        channels.append(data['channels'])
        intensities.append(data['intensities'])
        names.append(csv[0:-3] + 'cas')

    return pd.DataFrame(list(zip(channels, intensities, names)),
                        columns=['precise_channels', 'precise_intensities',
                                 'file_name'])


def get_isotope_data(path='../data/Elements.txt') -> pd.DataFrame:
    """
    Generate dataframe with isotope mass data for every element based on txt
    file with the information.
    """
    elements = []
    spots = []
    freqs = []
    with open(path) as file:
        for line in file.readlines():
            element_spots = []
            element_freqs = []
            sections = line.split('(')
            elements.append(sections[0].split()[2])
            for tup in sections[1:]:
                nums = tup.split(',')
                element_spots.append(float(nums[0]))
                element_freqs.append(float(nums[1].split(')')[0]))
            spots.append(element_spots)
            freqs.append(element_freqs)
    isotope_data = pd.DataFrame(list(zip(elements, spots, freqs)),
                                columns=['Element', 'Isotope Masses',
                                         'Isotope Frequencies'])
    hydrocarbs = get_hydrocarbs(51)
    df = pd.DataFrame(columns=['Element', 'Isotope Masses',
                               'Isotope Frequencies'])
    df.loc[0] = ['hydrocarbs', hydrocarbs, None]
    isotope_data = isotope_data.append(df)

    return isotope_data


def get_hydrocarbs(max_num_carbs) -> np.array:
    """
    Return array of hydrocarbon masses in amu by calculating all possible
    hydrocarbons with at most max_num_carbs carbons.
    """
    hydrocarbs = np.zeros(max_num_carbs)
    i = 0
    for c in range(1, max_num_carbs + 1):
        hydrocarbs[i] = (c * 12 + (2 * c + 1) * 1.00782)
        i += 1

    expanded_hydrocarbs = np.zeros(int(hydrocarbs[-2]))
    i = 0
    j = 0
    for hydrocarb in hydrocarbs:
        expanded_hydrocarbs[j] = hydrocarb
        j += 1
        if i < len(hydrocarbs) - 1:
            prev = hydrocarb
            diff = .01564
            dist = int(hydrocarbs[i + 1]) - int(hydrocarb) - 1
            for num in range(dist):
                mass = prev + 1 + diff / dist
                expanded_hydrocarbs[j] = mass
                j += 1
                prev = mass
        i += 1
    return expanded_hydrocarbs


def get_frags(loc='../data/Fragment Table.csv') -> pd.DataFrame:
    """
    Read in fragment table and return as an interpretable dataframe.
    """
    df = pd.read_csv(loc)
    a = list(df.columns)
    a[0] = float(a[0])
    a[-1] = int(a[-1])
    a[-2] = a[-2][0]
    df.columns = ['FragmentMass', 'FragmentLabel', 'Isotopes', 'Formula',
                  'FragmentID']
    b = {'FragmentMass': a[0], 'FragmentLabel': a[1], 'Isotopes': a[2],
         'Formula': a[3], 'FragmentID': a[4]}
    df = pd.concat([pd.DataFrame(b, index=[0]), df], sort=False)
    df.reset_index(drop=True, inplace=True)
    return df
