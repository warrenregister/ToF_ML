'''
DataFrame Generation and Data Transformation Functions
'''
from os import listdir

import numpy as np
import pandas as pd
from numba import njit


def add_error(number, modifier=2, tens=2, threshold=0.5, sub_one=False):
    '''
    Add or subtract a random amount of error based on the modifer and tens variables.
    Error calculated : (random num between 0 and 1 / modifier) / ten**tens
    '''
    randoms = np.random.rand(2)
    error = (randoms[0] / modifier) / 10**tens
    if sub_one:
        error = (1 - randoms[0] / modifier) / 10**tens
    new_num = 0
    mult = 1 if randoms[1] > threshold else -1
    new_num = number + mult * number * error
    return new_num, error * mult


def generate_data(data, tens, modifier, ranges=[0.2, 0.4, .6],
                  slope_cat=False, sub_one=False, slope_index=4):
    '''
    Takes calibrated ToF-MS data and adds error to the offset and mass.
    Returns multiclass classification dataset 0: no error,
    1: offset error, 2: slope error, 3: both. Does not recalibrate with new 
    slope and offset values

    Arguments -------
    data: dataframe to add error to
    tens: power of tens place for error
    modifier: modifier to apply to error, eg (rand num) / modifier
    ranges: percentile ranges for different outcomes slope error, offset
    and slope error, offset error
    slope_cat: if true both and slope error are in same the category of target.
    '''
    err_data = data.copy()
    target = []
    error_percent_slope = []
    error_percent_offset = []
    new_slope = []
    new_offset = []
    for row in err_data.itertuples():
        num = np.random.rand(1)
        if num < ranges[0]: #slope only
            target.append(2)
            slope, sl_err = add_error(row[slope_index], tens=tens, modifier=modifier, sub_one=sub_one)
            error_percent_slope.append(sl_err)
            new_slope.append(slope)
            new_offset.append(row.MassOffset)
            error_percent_offset.append(0)
        elif num >= ranges[0] and num < ranges[1]: # both
            if slope_cat:
                target.append(2)
            else:
                target.append(3)
            offset, off_err = add_error(row.MassOffset, tens=tens, modifier=modifier, sub_one=sub_one)
            error_percent_offset.append(off_err)
            new_offset.append(offset)
            slope, sl_err = add_error(row[slope_index], tens=tens, modifier=modifier, sub_one=sub_one)
            error_percent_slope.append(sl_err)
            new_slope.append(slope)
        elif num >= ranges[1] and num < ranges[2]: # offset only
            target.append(1)
            error_percent_slope.append(0)
            offset, off_err = add_error(row.MassOffset, tens=tens, modifier=modifier, sub_one=sub_one)
            error_percent_offset.append(off_err)
            new_offset.append(offset)
            new_slope.append(row[slope_index])
        else: # none
            target.append(0)
            error_percent_slope.append(0)
            error_percent_offset.append(0)
            new_slope.append(row[slope_index])
            new_offset.append(row.MassOffset)
    
        
    err_data['target'] = target
    err_data['err_prop_slope'] = error_percent_slope
    err_data['err_prop_offset'] = error_percent_offset
    err_data['Mass/Time'] = new_slope
    err_data['MassOffset'] = new_offset
    return err_data


@njit
def numba_mass_formula(channels: np.array, spec_bin_size, start_time,  mass_over_time, mass_offset):
    '''
    Fast conversion from flightime to mass. Uses numba.njit so run empty
    version of any function using this before the real run to compile in 
    njit.
    '''
    return ((channels * .001 * spec_bin_size + start_time) * mass_over_time + mass_offset)**2


def mass_formula(channels: np.array, spec_bin_size, start_time,  mass_over_time, mass_offset):
    '''
    Fast conversion from flightime to mass.
    '''
    return ((channels * .001 * spec_bin_size + start_time) * mass_over_time + mass_offset)**2


def generate_calibrated_data(data, slope_index=4, numba=False):
    '''
    Applies mass_formula to every row in dataset to allow
    calibrated graphs to be generated.
    '''
    new_data = data.copy()
    masses = []
    for row in new_data.itertuples():
        spec = row.SpecBinSize
        m_over_t = row[slope_index]
        m_offset = row.MassOffset
        time = row.StartFlightTime
        if not numba:
            masses.append(mass_formula(np.array(row.channels), spec, time,
                          m_over_t, m_offset))
        else:
            masses.append(numba_mass_formula(np.array(row.channels), spec,
                          time, m_over_t, m_offset))
    new_data['masses'] = masses
    return new_data


def get_precise_peaks(df, names):
    '''
    Return list of peaks channels and intensities from dataframe of precise peaks.
    '''
    all_peaks = []
    for row in df.iterrows():
        chans = row[1][names[0]][row[1][names[1]] > 0]
        intens = row[1][names[1]][row[1][names[1]] > 0]
        all_peaks.append(list(zip(chans, intens)))
    return all_peaks

def get_better_spectra(dir = '../data/SpectraCsvFiles_WatsonPeakFinder/'):
    '''
    Read in better peak info and trandform into lists of channels, intensities, names.
    '''
    channels = []
    intensities = []
    names = []
    for csv in listdir(dir):
        data = pd.read_csv(dir + csv, names=['channels', 'intensities'])
        channels.append(data['channels'])
        intensities.append(data['intensities'])
        names.append(csv[0:-3] + 'cas')
    
    return pd.DataFrame(list(zip(channels, intensities, names)),
     columns=['precise_channels', 'precise_intensities', 'file_name'])


def get_isotope_data(path='../data/Elements.txt'):
    '''
    Generate dataframe with isotope mass data for every element based on txt
    file with the information.
    '''
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
     columns=['Element', 'Isotope Masses', 'Isotope Frequencies'])
    hydrocarbs = get_hydrocarbs(51)
    df = pd.DataFrame(columns=['Element', 'Isotope Masses', 'Isotope Frequencies'])
    df.loc[0] = ['hydrocarbs', hydrocarbs, None]
    isotope_data = isotope_data.append(df)

    return isotope_data


def get_hydrocarbs(max_num_carbs):
    '''
    Return array of hydrocarbon masses in amu by calculating all possible
    hydrocarbons with at most max_num_carbs carbons.
    '''
    carbon = 12
    hydrogen = 1.00782
    hydrocarbs = np.zeros(max_num_carbs)
    i = 0
    for c in range(1, max_num_carbs + 1):
        hydrocarbs[i] = (c * 12 + (2 * c + 1) * 1.00782)
        i += 1

    expanded_hydrocarbs = np.zeros(int(hydrocarbs[-2]))
    i = 0
    j = 0
    for hydrocarb in hydrocarbs:
        expanded_hydrocarbs[j] = (hydrocarb)
        j += 1
        if i < len(hydrocarbs) - 1:
            prev = hydrocarb
            diff = .01564
            dist = int(hydrocarbs[i+1]) - int(hydrocarb) - 1
            for num in range(dist):
                mass = prev + 1 + diff / dist
                expanded_hydrocarbs[j] = (mass)
                j += 1
                prev = mass
        i += 1
    return expanded_hydrocarbs


@njit
def get_hydrocarbs_numba(max_num_carbs):
    '''
    Return array of hydrocarbon masses in amu by calculating all possible
    hydrocarbons with at most max_num_carbs carbons. Uses numba.njit
    to speed up additional runs, run get_hydrocarbs_numba(0) to compile the
    program so that numba can speed it up.
    '''
    carbon = 12
    hydrogen = 1.00782
    hydrocarbs = np.zeros(max_num_carbs)
    i = 0
    for c in range(1, max_num_carbs + 1):
        hydrocarbs[i] = (c * 12 + (2 * c + 1) * 1.00782)
        i += 1

    expanded_hydrocarbs = np.zeros(int(hydrocarbs[-2]))
    i = 0
    j = 0
    for hydrocarb in hydrocarbs:
        expanded_hydrocarbs[j] = (hydrocarb)
        j += 1
        if i < len(hydrocarbs) - 1:
            prev = hydrocarb
            diff = .01564
            dist = int(hydrocarbs[i+1]) - int(hydrocarb) - 1
            for num in range(dist):
                mass = prev + 1 + diff / dist
                expanded_hydrocarbs[j] = (mass)
                j += 1
                prev = mass
        i += 1
    return expanded_hydrocarbs



def get_ranges(mass_lists, length, max = 800):
    '''
    Computes no mans land spectras.
    '''
    ranges = [[x, x + 1] for x in range(length)]
    for masses in mass_lists:
        for mass in masses:
            i = int(mass)
            if mass < 235.043933:
                if round(mass) == i + 1 and mass < ranges[i][1]:
                    ranges[i][1] = mass
                elif round(mass) == i and mass > ranges[i][0]:
                    ranges[i][0] = mass
            else:
                ranges[i][0] = mass
                ranges[i][1] = i + .9871
    return ranges


def get_peak_suspiciousness(masses, ranges, show_correct_peaks=False, proportions=False, mass_thresh=800):
    '''
    Returns list of how suspicious peaks are, how far into no mans land they are.

    Arguments: -------
    masses: list of peak masses
    ranges: list of tuples representing 'No Peak Zones'
    show_correct_peaks: whether, if a peak is not in the 'No Peak Zone',
                        to show how far into the 'correct zone' it is.
    proportions: whether to show distance into zone by proportion or by actual
                 distance.
    mass_thresh: how far up the amu scale to find suspiciousness, default 800
    '''
    susses = []
    for mass in masses:
        if mass <= mass_thresh:
            range = ranges[int(mass)]
            val = min(abs(mass - range[0]), abs(mass - range[1]))
            if not proportions:
                if mass > range[0] and mass < range[1]:
                    susses.append(val)
                elif show_correct_peaks:
                    val = -1 * min(abs(mass - range[0]), abs(mass-range[1]))
                    susses.append(val)
                else:
                    susses.append(0)
            else:
                range_size = range[1] - range[0]
                if mass > range[0] and mass < range[1]:
                    susses.append(val / range_size)
                else: susses.append(0)

    return susses


def get_suspicious_peaks(masses, ranges, thresh=0.1):
    '''
    Returns all peaks with suspiciousness above threshold value
    '''
    susses = get_peak_suspiciousness(masses, ranges, True)
    a = np.array(masses)
    b = np.array(susses)
    return a[(b > thresh) & (a < 800)]


def get_frags(loc='../data/Fragment Table.csv'):
    '''
    Read in fragment table and return as an interpretable dataframe.
    '''
    df = pd.read_csv(loc)
    a = list(df.columns[0:])
    a[0] = float(a[0])
    a[-1] = int(a[-1])
    a[-2] = a[-2][0]
    df.columns = ['FragmentMass', 'FragmentLabel', 'Isotopes', 'Formula', 'FragmentID']
    b = {'FragmentMass': a[0], 'FragmentLabel': a[1], 'Isotopes':a[2] , 'Formula':a[3],'FragmentID':a[4]}
    df = pd.concat([pd.DataFrame(b, index=[0]), df], sort=False)
    df.reset_index(drop=True, inplace=True)
    return df


def get_frags_dists(masses, frags, thresh=0.003, ab=True):
    '''
    Determines which elemental / compound masses correspond
    to actual spectra masses and returns both the fragments
    and the distance between each fragment and its related mass in
    the given spectra.

    Arguments -------
    masses: list of masses for a spectrum
    frags: fragment list
    thresh: how close a fragment must be to a peak for it to be matched,
            default 3 mamu
    ab: whether to use absolute value for calculated distances, affects
        the average distance per spectrum.
    '''
    found_masses = []
    found_frags = []
    dists = []
    for mass in masses:
        not_found = True
        i = (len(frags)) // 2
        floor = 0
        cieling = len(frags) - 1

        def is_findable():
            if abs(floor - cieling) <= 1:
                return False
            return True
        
        while not_found:
            dist = frags[i] - mass
            if abs(dist) < thresh:
                not_found = False
                i = get_closest(i, frags, mass)
                found_masses.append(mass)
                found_frags.append(frags[i])
                if ab:
                    dists.append(abs(frags[i] - mass))
                else:
                    dists.append((frags[i] - mass))
            elif dist > 0:
                not_found = is_findable()
                cieling = i
                num = abs(floor - i)
                if num != 1:
                    i -= abs(floor - i) // 2
                else:
                    i -= 1
            else:
                not_found = is_findable()
                floor = i
                num = abs(cieling - i)
                if num != 1:
                    i += abs(cieling - i) // 2
                else:
                    i += 1
    return found_masses, found_frags, dists


def get_closest(i, frags, mass):
    '''
    Recursively checks that the closest fragment to a peak is selected.

    Arguments ------
    i: index in fragment list to start checking
    frags: list of mass fragments
    mass: mass of peak being matched
    '''
    d = abs(frags[i] - mass)
    if len(frags) > i + 1 and d > abs(frags[i + 1] - mass):
        i = get_closest(i + 1, frags, mass)
    elif i - 1 >= 0 and d > abs(frags[i - 1] - mass):
        i = get_closest(i - 1, frags, mass)
    return i


def get_calibration(data, dist_prop=.5, prop_thresh=0.65,
                    prop_col='adjusted_proportion_identified'):
    '''
    Generates calibration column using difference between avg distance from
    fragment to peak at 2 thresholds and the proportion of peaks matched at a
    low threshold.

    Arguments -------
    modifier: how much bigger 2nd avg dist can be before it throws calibration
              into question, e.g. .5 = 50% bigger
    prop_thresh: what proportion of matched fragments above which spectra are
                 considered calibrated
    '''
    calibs = []
    for tup in data.iterrows():
        row = tup[1]
        if row['diff'] < row['avg_dist_frags_low'] * dist_prop:
            if row[prop_col] > prop_thresh:
                calibs.append(1)
            else:
                calibs.append(0)
        else:
            calibs.append(0)
    return calibs


def get_fragment_stats(data, frag_loc=None, calib_diff_thresh=0.5,
                       calib_prop_thresh=0.55, threshs=[0.003, 0.007],
                       prop_name='proportion_identified'):
    '''
    Use fragment library to generate statistics which describe the calibration
    of TOF Mass Spectra.

    Arguments -------
    data: dataframe with spectra peak data
    frag_loc: optional file loc of fragment database
    calib_diff_thresh: threshold for diff value in determining calibration, see
                       get_calibration docstring
    calib_prop_thresh: threshold for prop value in determining calibration
    threshs: datastructure with 2 thresholds in amu for matching fragments at
             low and high thresholds.
    '''
    df = data.copy()
    frags = None
    if frag_loc:
        frags = get_frags(frag_loc)
    else:
        frags = get_frags()
    spots = frags['FragmentMass']
    dists_low_thresh = []
    dists_high_thresh = []
    nums = []
    props = []
    limited_props = []
    for row in df.itertuples():
        peaks = np.array(row.masses)
        masses, _, distances = get_frags_dists(peaks, spots, thresh=threshs[0])
        nums.append(len(masses))
        props.append(len(masses) / len(peaks))
        limited_props.append(len(masses) / len(peaks[peaks < 236]))
        if len(distances) > 0:
            dists_low_thresh.append(np.mean(distances))
        else:
            dists_low_thresh.append(0)
        _, _, distances = get_frags_dists(peaks, spots, thresh=threshs[1])
        if len(distances) > 0:
            dists_high_thresh.append(np.mean(distances))
        else:
            dists_high_thresh.append(0)
    df['avg_dist_frags_low'] = dists_low_thresh
    df['avg_dist_frags_high'] = dists_high_thresh
    df['adjusted_' + prop_name] = limited_props
    df[prop_name] = props
    df['diff'] = df['avg_dist_frags_high'] - df['avg_dist_frags_low']
    df['prop_diff_in_low'] = df['diff'] / df['avg_dist_frags_low']
    df['calibration'] = get_calibration(df, calib_diff_thresh,
                                        calib_prop_thresh, prop_col='adjusted_'+prop_name)
    return df

def recalibrate(row, spots, slope_amt=0, offset_amt=0, start_ind=2):
    '''
    Function which quickly asseses the calibration of a spectrum.
    '''
    slope = row[start_ind] + slope_amt * row[start_ind]
    offset = row[start_ind+1] + offset_amt * row[start_ind+1]
    peaks = []
    peaks = mass_formula(row.channels, row.SpecBinSize, row.StartFlightTime, slope,
                   offset)
    masses, _, distances = get_frags_dists(peaks, spots, thresh=.003)
    prop = len(masses) / len(peaks)
    #print('proportion matched: ' + str(prop))
    low_dist = 0
    if len(distances) > 0:
        low_dist = np.mean(distances)
    #print('distance low threshold: ' + str(low_dist))
    a, b, distances = get_frags_dists(peaks, spots, thresh=.007)
    high_dist = 0
    if len(distances) > 0:
        high_dist = np.mean(distances)
    #print('distance high threshold: ' + str(high_dist))
    return prop, low_dist, high_dist


def get_best_offset(spectrum, slope_range, offset_range, prev=0,
                    offsets=30, slopes=20, first=True, frags=None):
    '''
    Find best amount of slope/offset to add/subtract to slope/offset value
    to achieve the optimal calibration for spectrum. Calibration is measured
    using mass fragments. A spectrum which more matches to known masses is more
    calibrated than one with fewer. A spectrum whose matches are very close to
    known mass is more calibrated than one that is further away.

    Arguments -------
    spectrum: row from dataframe containing information on a spectrum
    slope_range: data structure containing min, max slope to try, slope erros
                 are typically smaller than offset errors.
    offset_range: data structure containing min, max offset agumentation
                       to try. This method shrinks the range iteratively until
                       the best offset is achieved.
    spots: a list of mass fragments
    '''
    if first:
        print('optimizing ' + spectrum.file_name)
        print('initial proportion: ' + str(spectrum.proportions_peaks_identified))
    if not frags:
        frags = get_frags()
    proportions = [0]
    low_distances = []
    high_distances = []
    best_prop = 0
    best_offset = spectrum.MassOffset
    best_slope = spectrum[2]
    best_ld = 1
    best_hd = 1
    ys = [] 
    i = 0
    mults = [1, -1]
    slope_space = np.linspace(slope_range[0], slope_range[1], slopes)
    while 1:
        slope = slope_space[i]
        for slope_mult in mults:
            slope_val = slope * slope_mult
            props = []
            low_dists = []
            high_dists = []
            y = []
            space = np.linspace(offset_range[0], offset_range[1], offsets)
            j = 0
            changed = False
            while 1:
                offset = space[j]
                for mult in mults:
                    improved = False
                    val = mult * offset
                    y.append(val)
                    prop, low, high = recalibrate(spectrum, frags, slope_val, val)
                    if prop > best_prop:
                        improved=True
                    elif (prop == best_prop and best_hd - high > 0 and
                          best_ld - low > 0):
                        improved = True
                    props.append(prop)
                    low_dists.append(low)
                    high_dists.append(high)
                    if improved:
                        best_prop = prop 
                        best_offset = val
                        best_slope = slope_val
                        best_ld = low
                        best_hd = high
                        edge = (np.where(space == mult * best_offset)[0][0] + .1)/ len(space)
                        edge = 2 * abs(.5 - edge)
                        slope_edge = np.where(slope_space == slope_mult * best_slope)[0][0]
                        slope_edge = 2 * abs(.5 - (slope_edge + .1) / len(slope_space))
                        changed = True            
                j += 1
                if j >= len(space):
                    break
            if changed:
                ys = y 
                proportions = props 
                low_distances = low_dists
                high_distances = high_dists
        i += 1
        if i >= len(slope_space):
            break
    print(best_prop)
    if best_prop > prev:
            a = best_offset - (.5/edge) * best_offset
            b = best_offset + (.5/edge) * best_offset
            c = best_slope + (.5 / slope_edge) * best_slope
            d = best_slope - (.5 / slope_edge) * best_slope
            p, ld, hd, yss, bp, bo, bs = get_best_offset(spectrum,
                                                            slope_range=[c,d],
                                                            offset_range=[a, b],
                                                            prev=best_prop,
                                                            offsets=20,
                                                            slopes=5, 
                                                            first=False, 
                                                            frags=frags)
            if bp >= best_prop:
                proportions = p
                low_distances = ld
                high_distances = hd
                ys = yss
                best_prop = bp
                best_offset = bo
                best_slope = bs
    return (proportions, low_distances, high_distances, ys, best_prop,
            best_offset, best_slope)