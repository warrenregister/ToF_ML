"""
DataFrame Generation and Data Transformation Functions
"""
from os import listdir

import numpy as np
import pandas as pd


def read_csvs(dir):
    """
    Read in all spectra csvs in the given directory to a dataframe containing
    values necessary for calibrating the spectrum.
    """
    info = ['Technique', 'StartFlightTime', 'Mass/Time', 'MassOffset',
            'SpecBinSize']
    data = [[] for name in info]
    channels = []
    intensities = []
    names = []
    for csv in listdir(dir):
        chans = []
        intens = []
        if csv[-3:] == 'csv':
            file = pd.read_csv(dir + csv, names=list('abcdefghijk'))
            names.append(csv)
            i = 0
            line = ''
            while 'EOFH' not in line:
                line = file.loc[i]['a']
                for j, name in enumerate(info):
                    if name in line:
                        datum = line.split(':')[-1]
                        try:
                            datum = float(datum)
                        except ValueError:
                            try:
                                datum = 0 if '-' in datum else 1
                                datum = int(datum)
                            except Exception as e:
                                print(e)
                        data[j].append(datum)
                i += 1
            for line in file.loc[i:].itertuples():
                try:
                    chans.append(float(line[1]))
                    intens.append(float(line[2]))
                except ValueError as e:
                    print(e)
                    print(line[1])
                    print(line[2])
            channels.append(chans)
            intensities.append(intens)
    return pd.DataFrame(
        list(zip(names, data[1], data[2], data[3], data[4], data[0],
                 channels, intensities)),
        columns=['file_name', 'StartFlightTime',
                 'MassOverTime', 'MassOffset', 'SpecBinSize',
                 'Technique', 'channels', 'intensities'])


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

