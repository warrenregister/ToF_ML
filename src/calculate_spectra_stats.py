"""
Methods for calculating stats which describe the calibration of Time Of Flight
Mass Spectra.
"""
import numpy as np
import pandas as pd
from data_generation import get_frags, mass_formula, get_isotope_data


def get_ranges(mass_lists: pd.Series, mass_limit: int) -> list:
    """
    Creates list of No Peak Zones from 1 Dalton to mass_limit Daltons.

    Returns list of NPZs between each nominal mass up until the mass_limit.

    Arguments -------
    mass_lists: a pd.Series containing lists or numpy arrays of the
    mass values for a set of TOF Spectra.
    mass_limit: int representing how far up the dalton/amu scale to generate no
    peak zones.
    """
    ranges = [[x, x + 1] for x in range(mass_limit)]
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


def get_tallest_per_nominal_mass(masses, intensities) -> tuple:
    """
    Given a list of masses and a list of their corresponding intensities
    finds the tallest peak at each whole number or nominal mass.

    Returns 2 lists containing the tallest masses at each nominal mass and
    their corresponding intensities / occurrence counts.

    Arguments -------
    masses: List of mass values for a TOF Spectrum
    intensities: list of occurrence counts / intensities for a TOF Spectrum
    """

    if len(masses) != len(intensities) :
        print("Invalid Argument Error, mass and intensity lists are of" +
              "unequal length.")
        return masses, intensities

    masses = [x for x, y in sorted(zip(masses, intensities))]
    intensities = [y for x, y in sorted(zip(masses, intensities))]

    new_masses = []
    new_intensities = []
    curr_nom_mass = 0
    max_height_curr_nom_mass = 0
    max_index = None
    for i, mass in enumerate(masses):
        if round(mass) == curr_nom_mass:
            if intensities[i] > max_height_curr_nom_mass:
                max_height_curr_nom_mass = intensities[i]
                max_index = i
        else:
            if max_height_curr_nom_mass != 0:
                new_masses.append(masses[max_index])
                new_intensities.append(intensities[max_index])

            curr_nom_mass = round(mass)
            max_height_curr_nom_mass = intensities[i]
            max_index = i
    new_masses.append(masses[max_index])
    new_intensities.append(intensities[max_index])

    return new_masses, new_intensities


def get_distance_npz(masses, ranges, show_correct_peaks=False,
                     proportions=False, mass_thresh=800) -> list:
    """
    Returns list of how how far into the No Peak Zone the given masses are.

    Returns list of distances into NPZ if proportions is False otherwise returns
    list of proportionally how far towards the center of the NPZ each peak is.

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


def get_suspicious_peaks(masses, ranges, thresh=0.1, mass_limit=800) -> tuple:
    """
    Returns list of all peaks with distance / proportion into the No Peak Zone
    above the given threshold, as well as the mean distance into the
    No Peak Zone.

    Returns all suspicious peaks as well as the average suspicion level of all
    passed in peaks.

    Arguments -------
    masses: list of peak mass values.
    ranges: list of tuples representing no peak zones.
    thresh: threshold beyond which peaks in NPZ are suspicious, default .1
    mass_limit: mass limit beyond which npz is not checked, default 800
    """
    susses = get_distance_npz(masses, ranges, False)
    a = np.array(masses)
    a = a[a < mass_limit]
    b = np.array(susses)
    return a[(b > thresh)], np.mean(susses)


def calc_npz_score(masses, intensities, ranges, thresh=0.1) -> float:
    """
    Modified version of get suspicious peaks which calculates an NPZ score
    which weights peaks based on their height.

    Returns NPZ score.

    Arguments -------
    masses: list of peak mass values.
    intensities: list of peak intensity values
    ranges: list of tuples representing no peak zones.
    thresh: threshold beyond which peaks in the No Peak Zone are suspicious.
    """
    susses = get_distance_npz(masses, ranges, False)
    masses = np.array(masses)
    masses = masses[masses < 800]
    intens = np.array(intensities)
    intens = intens[masses < 800]
    b = np.array(susses)

    peaks = masses[(b > thresh)]
    intens =intens[(b > thresh)]
    sum = 0
    for i, peak in enumerate(peaks):
        sum += intens[i]

    return sum / np.sum(intensities)


def calc_npz_stats(masses, intensities, ranges, mass_limit=800) -> tuple:
    """
    Calculates and returns number of tallest peaks in the NPZ, proportion of
    tallest peaks in the NPZ, and the avg distance into the NPZ.

    Returns number of npz peaks, proportion of npz peaks, and avg dist into NPZ.

    Arguments -------
    masses: list of mass values for a spectrum
    intensities: list of intensities for a spectrum's masses
    ranges: list of NPZ ranges
    mass_limit: threshold beyond which ragnes are not calculated / included
    """
    peaks, heights = get_tallest_per_nominal_mass(masses, intensities)
    sus_peaks, ad = get_suspicious_peaks(peaks, ranges, mass_limit=mass_limit)
    return len(sus_peaks), len(sus_peaks) / (len(peaks) + .001), ad


def calc_fragment_matches(masses, frags, threshes=(0.003, 0.007),
                          ab=True, index=False) -> tuple:
    """
    Determines which elemental / compound masses correspond
    to actual spectra masses and returns both the fragments
    and the distance between each fragment and its related mass in
    the given spectra.

    Returns a set of values for each threshold in threshes. These values
    are a list of masses matched, a list of the corresponding fragments, and a
    list of their separations. If index is True also returns a list of indexes
    for each returned mass in the passed in masses list.

    Arguments -------
    masses: list of masses for a spectrum
    frags: fragment list
    thresh: how close a fragment must be to a peak for it to be matched,
    default .003 dalton/amu
    ab: whether to use absolute value for calculated distances, affects
        the average distance per spectrum.
    """
    stats = [[[] for x in range(3)] for y in range(len(threshes))]
    if index:
        stats = [[[] for x in range(4)] for y in range(len(threshes))]
    max_frag = max(frags) + max(threshes)
    findable_masses = np.array(masses)
    findable_masses = findable_masses[findable_masses < max_frag]
    for mass in findable_masses:
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
                        if index:
                            stats[j][3].append(masses.index(mass))
            elif dist > 0:  # positive dist
                not_found = is_findable()
                ceiling = i
                num = abs(floor - i)
                if num != 1:
                    i -= abs(floor - i) // 2
                else:
                    i -= 1
            else:  # negative dist
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

    Returns index of closest fragment.

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


def calc_spectrum_stats(data, frag_loc, iso_loc, threshes=(0.003, 0.007),
                        inplace=False):
    """
    Use fragment library and the No Peak Zone to calculate descriptive
    statistics for a dataset.

    Returns a new dataframe with added columns for the stats if inplace is
    False, otherwise returns nothing because original dataframe is changed
    instead.

    Arguments -------
    data: dataframe with spectra peak data
    frag_loc: optional file loc of fragment database
    threshes: data structure with 2 thresholds in amu for matching fragments at
    low and high thresholds.
    """
    if not inplace:
        df = data.copy()
    else:
        df = data
    try:
        frags = get_frags(frag_loc)['FragmentMass']
    except FileNotFoundError:
        raise Exception('Invalid frag_loc, file not found.')
    try:
        ranges = get_ranges(get_isotope_data(iso_loc)['Isotope Masses'], 800)
    except FileNotFoundError:
        raise Exception("Invalid iso_loc, file not found.")

    dists_low_thresh = []
    stds_low_thresh = []
    dists_high_thresh = []
    stds_high_thresh = []
    nums_matches = []
    props_matched = []
    nums_npz = []
    props_npz = []
    avg_dists_npz = []
    for row in df.itertuples():
        peaks = np.array(row.masses)
        masses, _, dist1, _, _, dist2 = calc_fragment_matches(peaks, frags,
                                                              threshes)
        limited_prop = len(masses) / (len(peaks[peaks < 236]) + .001)

        low_dist = 0
        std_low_dist = 0
        high_dist = 0
        std_high_dist = 0
        if len(dist1) > 0:
            low_dist = np.mean(dist1)
            std_low_dist = np.std(dist1)
        if len(dist2) > 0:
            high_dist = np.mean(dist2)
            std_high_dist = np.std(dist2)

        dists_high_thresh.append(high_dist)
        stds_high_thresh.append(std_high_dist)
        dists_low_thresh.append(low_dist)
        stds_low_thresh.append(std_low_dist)
        nums_matches.append(len(masses))
        props_matched.append(limited_prop)

        num, prop, ad = calc_npz_stats(row.masses, row.intensities, ranges, 800)
        nums_npz.append(num)
        props_npz.append(prop)
        avg_dists_npz.append(ad)
    df['Avg Fragment Separation Low Thresh'] = dists_low_thresh
    df['STD Fragment Sep Low Thresh'] = stds_low_thresh
    df['Avg Fragment Separation High Thresh'] = dists_high_thresh
    df['STD Fragment Sep High Thresh'] = stds_high_thresh
    df['Adjusted Proportion Matched'] = props_matched
    df['Number Matched'] = nums_matches
    df['Number Tallest NPZ'] = nums_npz
    df['Proportion Tallest NPZ'] = props_npz
    df['Avg Dist Tallest NPZ'] = avg_dists_npz

    if not inplace:
        return df
    return None


def calc_new_spectrum_stats(row, spots, slope_val, offset_val, ranges,
                            threshes=(.003, .007), add=False) -> tuple:
    """
    Changes the calibration of a spectrum and then calculates descriptive
    statistics for the new version of the spectrum. Can change spectrum
    calibration in a number of ways using different types of slope_val and
    offset_val. Can be treated as proportions of slope/offset to add to original
    values, or a number to add to slope/offset.


    Returns proportion matched, number of peaks matched, low threshold fragment
    separation, high threshold fragment separation, proportion of tallest peaks
    in the NPZ, number of tallest peaks in the NPZ, avg dist into the NPZ.

    Arguments -------
    row: row corresponding to spectra
    spots: pd.Series of fragments
    slope_val: either a proportion of slope to augment slope with or a new value
    for slope.
    offset_val: either a proportion of offset to augment offset with or a new
    value for offset.
    ranges: list of tuples representing No Peak Zones.
    threshes: (Optional) data structure containing threshold values to check
    for fragment matches under
    add: (Optional) True if adding slope/offset val to original False if using
    slope/offset val as a proportion of original to change original by.
    mass_limit: (Optional) mass limit used to create ranges list, default 800
    """
    slope = row.MassOverTime + slope_val * row.MassOverTime
    offset = row.MassOffset + offset_val * row.MassOffset
    if add:
        slope = row.MassOverTime + slope_val
        offset = row.MassOffset + offset_val

    peaks = mass_formula(np.array(row.channels), row.SpecBinSize,
                         row.StartFlightTime, slope, offset)
    masses, _, dist1, masses2, frags2, dist2 = calc_fragment_matches(peaks,
                                                                     spots,
                                                                     threshes)
    num_npz, prop_npz, ad_npz = calc_npz_stats(peaks, row.intensities, ranges)

    prop = len(masses) / (len(peaks[peaks < 236]) + 1)
    low_dist = 0
    high_dist = 0
    if len(dist1) > 0:
        low_dist = np.mean(dist1)
    if len(dist2) > 0:
        high_dist = np.mean(dist2)

    return prop, len(masses), low_dist, high_dist, prop_npz, num_npz, ad_npz


def recalibrate(peak1, flight_time1, loc, flight_time2) -> tuple:
    """
    Recalibrate a spectra so that 1 peak stays where it is and another is
    shifted to a desired position.

    Returns new slope and offset values.

    Arguments ------
    peak1: peak mass value, this one stays the same
    flight_time1: peak1's corresponding flight time
    loc: mass value to shift peak2 towards
    flight_time2: peak2's corresponding flight time
    """
    peak1 = np.sqrt(peak1)
    loc = np.sqrt(loc)
    slope = (peak1 - loc) / (flight_time1 - flight_time2)
    offset = slope * (0 - flight_time1) + peak1
    return slope, offset


def simulate_calibration_mistake(masses, channels, slope, offset,
                                 bin_size, start_time, changes=False) -> tuple:
    """
    In order to create more realistic errors, this function searches spectra for
    good matches nearby other peaks, chooses sets that are relatively far apart
    and recalibrates so that one peak is switched with a nearby one.

    Returns either new slope and offset value if changes is False, or the
    difference between the new and old values if changes is True.

    Arguments -------
    masses: list of peak masses
    channels: list of channels associated with masses
    slope: original slope value
    offset: original offset value
    bin_size: spectrum bin size
    start_time: start flight time
    changes: (Optional) if True returns changes to slope offset instead of new
    values, default False
    """
    arr = np.array(masses)
    peak1 = arr[arr>11][0]
    time1 = channels[masses.index(peak1)] * .001 * bin_size + start_time
    peak2 = arr[arr > peak1 + 20][0]
    mult = -1 if np.random.randint(0, 10, 1) <= 4 else 1
    e = np.random.uniform(.01, .0015, 1)[0] * mult
    time2 = (np.sqrt(peak2 + e) - offset) / slope
    new_slope, new_offset = recalibrate(peak1, time1, peak2, time2)
    if changes:
        return new_slope - slope, new_offset - offset
    return new_slope, new_offset
