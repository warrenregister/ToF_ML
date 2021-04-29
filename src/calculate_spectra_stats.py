"""
Methods for calculating stats which describe the calibration of Time Of Flight
Mass Spectra.
"""
import numpy as np
import pandas as pd
from data_generation import get_frags, mass_formula, get_isotope_data


def get_ranges(mass_lists: pd.Series, length: int) -> list:
    """
    Creates list of No Peak Zones from 1 Dalton to length Daltons.

    Arguments -------
    mass_lists: a pd.Series containing lists or numpy arrays of the
    mass values for a set of TOF Spectra.
    length: int representing how far up the dalton/amu scale to generate no
    peak zones.
    """
    ranges = [[x, x + 1] for x in range(length)]
    for masses in mass_lists:
        for mass in masses:
            i = int(mass)
            if mass < 236:
                if round(mass) == i + 1 and mass < ranges[i][1]:
                    ranges[i][1] = mass
                elif round(mass) == i and mass > ranges[i][0]:
                    ranges[i][0] = mass
            else:
                ranges[i][0] = mass
                ranges[i][1] = i + .9871
    return ranges


def get_distance_npz(masses, ranges, show_correct_peaks=False,
                     proportions=False, mass_thresh=800) -> list:
    """
    Returns list of how how far into the No Peak Zone the given masses are.

    Arguments: -------
    masses: list of peak masses
    ranges: list of tuples representing 'No Peak Zones'
    show_correct_peaks: whether, if a peak is not in the 'No Peak Zone',
    to show how far into the 'correct zone' it is.
    proportions: boolean, whether to show distance into zone by proportion
    instead of distance.
    mass_thresh: how far up the amu scale to find the distance into NPZ,
    default 800
    """
    dists_or_props = []
    for mass in masses:
        if mass <= mass_thresh:
            zone = ranges[int(mass)]
            val = min(abs(mass - zone[0]), abs(mass - zone[1]))
            if not proportions:
                if zone[0] < mass < zone[1]:
                    dists_or_props.append(val)
                elif show_correct_peaks:
                    val = -1 * min(abs(mass - zone[0]), abs(mass - zone[1]))
                    dists_or_props.append(val)
                else:
                    dists_or_props.append(0)
            else:
                range_size = zone[1] - zone[0]
                if zone[0] < mass < zone[1]:
                    dists_or_props.append(val / range_size)
                else:
                    dists_or_props.append(0)

    return dists_or_props


def get_suspicious_peaks(masses, ranges, thresh=0.1) -> np.array:
    """
    Returns list of all peaks with distance / proportion into the No Peak Zone
    above the given threshold, thresh.

    Arguments -------
    masses: list of peak mass values.
    ranges: list of tuples representing no peak zones.
    thresh: threshold beyond which peaks in the No Peak Zone are suspicious.
    """
    susses = get_distance_npz(masses, ranges, True)
    a = np.array(masses)
    a = a[a < 800]
    b = np.array(susses)
    return a[(b > thresh)]


def get_calibration(differences, avg_dist_low, proportions, dist_prop=.5,
                    prop_thresh=0.65) -> pd.Series:
    """
    Generates 'calibration' column using difference between avg distance from
    fragment to peak at 2 thresholds and the proportion of peaks matched at a
    low threshold, typically .003 daltons/amu.

    differences, avg_dist_low, proportions must all be the same length.

    Arguments -------
    differences: pd.Series showing difference between avg distance from fragment
    at a low and high threshold divided by the low threshold value.
    avg_dist_low: pd.Series of avg distance from fragment at a low threshold,
    typically .003 daltons/amu.
    proportions: pd.Series of proportions of peaks matched to a fragment at a
    low threshold of .003 daltons/amu.
    modifier: how much bigger 2nd avg dist can be before it throws calibration
    into question, e.g. .5 = 50% bigger
    prop_thresh: what proportion of matched fragments above which spectra are
    considered calibrated
    """
    if (len(differences) != len(proportions) and
            len(proportions) != len(avg_dist_low)):
        raise Exception("differences, proportions, and avg_dist_low must" +
                        "all be the same length.")

    calibrations = []
    for index in range(len(proportions)):
        if differences[index] < avg_dist_low[index] * dist_prop:
            if proportions[index] > prop_thresh:
                calibrations.append(1)
            else:
                calibrations.append(0)
        else:
            calibrations.append(0)
    return pd.Series(calibrations)


def calc_fragment_matches(masses, frags, threshes=(0.003, 0.007),
                          ab=True) -> tuple:
    """
    Determines which elemental / compound masses correspond
    to actual spectra masses and returns both the fragments
    and the distance between each fragment and its related mass in
    the given spectra.

    Arguments -------
    masses: list of masses for a spectrum
    frags: fragment list
    thresh: how close a fragment must be to a peak for it to be matched,
            default .003 dalton/amu
    ab: whether to use absolute value for calculated distances, affects
        the average distance per spectrum.
    """
    stats = [[[] for x in range(3)] for y in range(len(threshes))]
    for mass in masses:
        not_found = True
        i = (len(frags)) // 2
        floor = 0
        ceiling = len(frags) - 1

        def is_findable():
            if abs(floor - ceiling) <= 1:
                return False
            return True

        while not_found:
            dist = frags[i] - mass
            if abs(dist) < max(threshes):
                not_found = False
                i = get_closest(i, frags, mass)
                dist = frags[i] - mass
                for j, thresh in enumerate(threshes):
                    if abs(dist) < threshes[j]:
                        stats[j][0].append(mass)
                        stats[j][1].append(frags[i])
                        if ab:
                            stats[j][2].append(abs(frags[i] - mass))
                        else:
                            stats[j][2].append((frags[i] - mass))
            elif dist > 0:
                not_found = is_findable()
                ceiling = i
                num = abs(floor - i)
                if num != 1:
                    i -= abs(floor - i) // 2
                else:
                    i -= 1
            else:
                not_found = is_findable()
                floor = i
                num = abs(ceiling - i)
                if num != 1:
                    i += abs(ceiling - i) // 2
                else:
                    i += 1
    return tuple([a for x in stats for a in x])


def get_closest(i, frags, mass) -> int:
    """
    Recursively checks that the closest fragment to a peak is selected.

    Arguments ------
    i: index in fragment list to start checking
    frags: list of mass fragments
    mass: mass of peak being matched
    """
    d = abs(frags[i] - mass)
    if len(frags) > i + 1 and d > abs(frags[i + 1] - mass):
        i = get_closest(i + 1, frags, mass)
    elif i - 1 >= 0 and d > abs(frags[i - 1] - mass):
        i = get_closest(i - 1, frags, mass)
    return i


def get_fragment_stats(data, frag_loc=None, difference_thresh=0.5,
                       proportion_thresh=0.55, threshs=(0.003, 0.007),
                       prop_name='proportion_identified') -> pd.DataFrame:
    """
    Use fragment library to generate statistics which describe the calibration
    of TOF Mass Spectra.

    Arguments -------
    data: dataframe with spectra peak data
    frag_loc: optional file loc of fragment database
    difference_thresh: threshold for diff value in determining calibration, see
    get_calibration docstring
    proportion_thresh: threshold for prop value in determining calibration
    threshs: data structure with 2 thresholds in amu for matching fragments at
    low and high thresholds.
    """
    df = data.copy()
    if frag_loc:
        frags = get_frags(frag_loc)
    else:
        frags = get_frags()
    frags = frags['FragmentMass']
    dists_low_thresh = []
    dists_high_thresh = []
    nums = []
    props = []
    limited_props = []
    for row in df.itertuples():
        peaks = np.array(row.masses)
        masses, _, dist1, _, frags2, dist2 = calc_fragment_matches(peaks,
                                                                   frags,
                                                                   threshs)
        prop = len(masses) / (len(peaks) + .001)
        lprop = len(masses) / (len(peaks[peaks < 236]) + .001)
        low_dist = 0
        high_dist = 0
        if len(dist1) > 0:
            low_dist = np.mean(dist1)
        if len(dist2) > 0:
            high_dist = np.mean(dist2)
        dists_high_thresh.append(high_dist)
        dists_low_thresh.append(low_dist)
        nums.append(len(masses))
        props.append(prop)
        limited_props.append(lprop)
    df['avg_dist_frags_low'] = dists_low_thresh
    df['avg_dist_frags_high'] = dists_high_thresh
    df['adjusted_' + prop_name] = limited_props
    df[prop_name] = props
    df['diff'] = df['avg_dist_frags_high'] - df['avg_dist_frags_low']
    df['prop_diff_in_low'] = df['diff'] / df['avg_dist_frags_low']
    df['calibration'] = get_calibration(df['diff'], df['avg_dist_frags_low'],
                                        df['adjusted_' + prop_name],
                                        difference_thresh,
                                        proportion_thresh)
    return df


def calc_new_spectrum_stats(row, spots, slope_val, offset_val, augment=True,
                            num_peaks=False, threshes=(.003, .007), loc=None,
                            ranges=None) -> tuple:
    """
    Calculates % of peaks matched to fragments, avg distance from peak to
    fragment at a low and high threshold, how many peaks are in the no peak
    zone, their avg distance into the npz, and optionally how many peaks were
    matched to fragments.

    Arguments -------
    row: row corresponding to spectra
    spots: pd.Series of fragments
    slope_val: either a proportion of slope to augment slope with or a new value
    for slope.
    offset_val: either a proportion of offset to augment offset with or a new
    value for offset.
    augment: (Optional) whether to treat slope_val/offset_val as a proportion
    or value
    num_peaks: (Optional) if true also returns the number of peaks of matched
    per spectrum
    threshes: (Optional) data structure containing threshold values to check
    for fragment matches under
    loc: (Optional) string, path to Isotope Data file
    ranges: (Optional) string, list of tuples representing No Peak Zones.
    """
    if augment:
        slope = row.MassOverTime + slope_val * row.MassOverTime
        offset = row.MassOffset + offset_val * row.MassOffset
    else:
        slope = slope_val
        offset = offset_val

    peaks = mass_formula(np.array(row.channels), row.SpecBinSize,
                         row.StartFlightTime, slope, offset)
    masses, _, dist1, masses2, frags2, dist2 = calc_fragment_matches(peaks,
                                                                     spots,
                                                                     threshes)
    num_npz, ad_npz = calc_npz_peaks(peaks, loc, ranges)

    prop = len(masses) / (len(peaks[peaks < 236]) + 1)
    low_dist = 0
    high_dist = 0
    if len(dist1) > 0:
        low_dist = np.mean(dist1)
    if len(dist2) > 0:
        high_dist = np.mean(dist2)
    if num_peaks:
        return prop, low_dist, high_dist, num_npz, ad_npz, len(masses)
    else:
        return prop, low_dist, high_dist, num_npz, ad_npz


def calc_npz_peaks(masses: pd.Series, loc:str = None,
                   ranges: list = None) -> tuple:
    """
    Calculates the number of peaks that are in the No Peak Zones defined by
    ranges as well as their average distance into those zones.

    Arguments -------
    masses: data structure containing the mass of all peaks in a spectrum
    loc: (Optional) string, path to isotope mass database
    ranges: (Optional) list of No Peak Zones if not passed, calls
    get_ranges(masses, 2000) to generate ranges.
    """
    if ranges is None:
        try:
            if loc is None:
                ranges = get_ranges(get_isotope_data()['Isotope Masses'], 2000)
            else:
                ranges = get_ranges(get_isotope_data(loc)['Isotope Masses'],
                                    2000)
        except FileNotFoundError as e:
            if loc is not None:
                print('Error: Isotope Data not found at loc:' + loc)
            else:
                print('Error: please specify where Isotope Data file is.')
            raise e

    peaks = get_suspicious_peaks(masses, ranges)
    avg_dist = get_distance_npz(masses, ranges)
    return len(peaks), np.mean(avg_dist)