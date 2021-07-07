"""
Contains Methods for extracting features from raw TOF Spectra data.
"""
import numpy as np
from calculate_spectra_stats import calc_npz_stats, calc_fragment_matches


def get_split_npz_stats(masses, intensities, ranges, mass_limit, splits):
    """
    Calculates NPZ Stats for subsections of the total NPZ
    area (0-~700 amu). Each range's stats are either calculated with amu or with
    ppm for distance into NPZ depending on the values in ppm_statuses for each
    split. Number of sections determined by num_splits, specific ranges per
    split can be specified with split_ranges.

    Arguments -------
    masses: list of masses for a spectrum
    intensities: list of intensities for a spectrum
    ranges: npz ranges
    splits: (Optional) list upper bounds for each split, if not passed
    splits each range evenly
    """
    if splits is not None:
        if max(splits) != mass_limit:
            raise ValueError(
                "split_ranges must have length equal to mass_limit")
        splits = splits

    results = []
    lower_split_bound = 0
    masses_arr = np.array(masses)
    inten_arr = np.array(intensities)
    for i in range(len(splits)):
        set_masses = masses_arr[(masses_arr >= lower_split_bound) &
                                (masses_arr < splits[i])].copy()
        set_masses = list(set_masses)
        set_intens = inten_arr[(masses_arr >= lower_split_bound) &
                               (masses_arr < splits[i])].copy()
        set_intens = list(set_intens)
        num, prop, ad = calc_npz_stats(set_masses, set_intens, ranges,
                                       717)
        results.append([num, prop, ad])
        lower_split_bound = splits[i]
    return results


def get_split_fragment_stats(num_splits, ppm_list, masses, frags,
                             threshes=(.003, .007), split_ranges=None):
    """
    Calculates Fragment Stats for subsections of the total Fragment Matching
    area (0-235 amu). Each range's stats are either calculated with amu or with
    ppm for separations depending on the values in ppm_statuses for each split.
    Number of sections determined by num_splits, specific ranges per split can
    be specified with split_ranges.

    Arguments -------
    num_splits: number of sections to split amu from 0-235 into, defaults to
    splitting them evenly unless split_ranges are passed.
    ppm_list: list of booleans for whether to calculate separation stats using
    ppm (True) or amu (False)
    masses: list of masses for a spectrum
    frags: list of all known frags
    threshes: (Optional) threshold values for get_fragment_matches
    split_ranges: (Optional) list upper bounds for each split, if not passed
    splits each range evenly
    """
    if split_ranges is not None:
        if len(split_ranges) != num_splits:
            raise ValueError(
                "split_ranges must have length equal to num_splits")
        ranges = split_ranges
    else:
        step_size = max(frags) / num_splits
        ranges = [x * step_size for x in range(1, num_splits + 1)]

    results = []
    lower_bound = 0
    masses_arr = np.array(masses)
    for i in range(num_splits):
        set_masses = masses_arr[(masses_arr >= lower_bound) &
                                (masses_arr < ranges[i])]
        values = calc_fragment_matches(set_masses, frags, threshes,
                                       ppm=ppm_list[i])
        matches, _, dists1, _, _, dists2 = values
        dist1 = np.mean(dists1) if len(dists1) > 0 else 0
        dist2 = np.mean(dists2) if len(dists2) > 0 else 0
        result = [len(matches), len(matches) / (len(set_masses) + .01),
                  dist1, dist2]
        results.append(result)
        lower_bound = ranges[i]
    return results


def get_isotope_stats(masses, intensities, isotope_finder):
    """
    Calculate isotope stats for a single spectra

    Arguments -------
    masses: list of spectrum masses
    intensities: list of spectrum intensities
    apr: Atomic Pattern Recognizer object
    """
    elements = isotope_finder.find_atomic_patterns(masses, intensities)
    avg_dist_three = 0
    avg_dist_two = 0
    avg_abund_sep_three = 0
    avg_abund_sep_two = 0
    two_elems = 0
    three_elems = 0
    two_dists = []
    three_dists = []
    two_abundance_seps = []
    three_abundance_seps = []
    for key in elements.keys():
        if len(elements[key][2]) == 2:
            two_elems += 1
            two_dists.append(elements[key][0])
            if elements[key][1] != -10 or elements[key][1] != 10000:
                two_abundance_seps.append(elements[key][1])
        elif len(elements[key][2]) >= 3:
            three_elems += 1
            three_dists.append(elements[key][0])
            three_abundance_seps.append(elements[key][1])
    if len(three_dists) > 0:
        avg_dist_three = np.mean(three_dists)
    else:
        avg_dist_three = 0
    if len(three_abundance_seps) > 0:
        three_abundance_seps = np.mean(three_abundance_seps)
    else:
        three_abundance_seps = 0
    if len(two_dists) > 0:
        avg_dist_two = np.mean(two_dists)
    else:
        avg_dist_two = 0
    if len(two_abundance_seps) > 0:
        two_abundance_seps = np.mean(two_abundance_seps)
    else:
        two_abundance_seps = 0
    return (two_elems, avg_dist_two, two_abundance_seps, three_elems,
            avg_dist_three, three_abundance_seps)


def get_isotope_columns(data, features, isotope_finder, inplace=False):
    """
    Populate a feature dataframe with Isotope features calculated from
    a dataframe of raw spectra data.

    Arguments -------
    data: dataframe of raw spectra data
    features: dataframe of corresponding features to add new ones too
    isotope_finder: IsotopeFinder object
    """
    if inplace:
        df = features
    else:
        df = features.copy()
    num_2_elems = []
    sep_2_elems = []
    sep_2_abunds = []
    num_3_elems = []
    sep_3_elems = []
    sep_3_abunds = []
    for row in data.itertuples():
        print(row.Index, row.file_name)
        curr_mass = 0
        index = 0
        while curr_mass < 236 and index < len(row.masses):
            curr_mass = row.masses[index]
            index += 1
        if index < 800:
            a, b, c, d, e, f = get_isotope_stats(row.masses,
                                                     row.intensities,
                                                     isotope_finder)
            num_2_elems.append(a)
            sep_2_elems.append(b)
            sep_2_abunds.append(c)
            num_3_elems.append(d)
            sep_3_elems.append(e)
            sep_3_abunds.append(f)
        else:
            num_2_elems.append(0)
            sep_2_elems.append(0)
            sep_2_abunds.append(0)
            num_3_elems.append(0)
            sep_3_elems.append(0)
            sep_3_abunds.append(0)

    df['Num 2 Isotope Elements'] = num_2_elems
    df['Separation 2 Isotope Elements'] = sep_2_elems
    df['Abundance Sep 2 Isotope Elements'] = sep_2_abunds
    df['Num 3 Isotope Elements'] = num_3_elems
    df['Separation 3 Isotope Elements'] = sep_3_elems
    df['Abundance Sep 3 Isotope Elements'] = sep_3_abunds
    if not inplace:
        return df
