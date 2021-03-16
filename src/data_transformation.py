'''
DataFrame Generation and Data Transformation Functions
'''
import numpy as np
from os import listdir
import pandas as pd
from scipy.signal import find_peaks
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA




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
    if randoms[1] < threshold:
        new_num  = number - number * error
        error = error * -1
    else:
        new_num = number + number * error
    return new_num, error


def generate_data(data, tens, modifier, use_ranges=False, ranges=[0.2, 0.4, .6], slope_cat=False, sub_one=False):
    '''
    Takes calibrated ToF-MS data and adds error to the offset and mass. If use_ranges returns
    multiclass classification dataset 0: no error, 1: offset error, 2: slope error, 3: both.

    Arguments -------
    data: dataframe to add error to
    tens: power of tens place for error
    modifier: modifier to apply to error, eg (rand num) / modifier
    use_ranges: boolean if true returns multiclass dataset for combinations of error.
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
        if use_ranges:
            if num < ranges[0]: #slope only
                target.append(2)
                slope, sl_err = add_error(row[4], tens=tens, modifier=modifier, sub_one=sub_one)
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
                slope, sl_err = add_error(row[4], tens=tens, modifier=modifier, sub_one=sub_one)
                error_percent_slope.append(sl_err)
                new_slope.append(slope)
            elif num >= ranges[1] and num < ranges[2]: # offset only
                target.append(1)
                error_percent_slope.append(0)
                offset, off_err = add_error(row.MassOffset, tens=tens, modifier=modifier, sub_one=sub_one)
                error_percent_offset.append(off_err)
                new_offset.append(offset)
                new_slope.append(row[4])
            else:
                target.append(0)
                error_percent_slope.append(0)
                error_percent_offset.append(0)
                new_slope.append(row[4])
                new_offset.append(row.MassOffset)
        else:
            if num < 0.5:
                target.append(0)
                offset, off_err = add_error(row.MassOffset, tens=tens, modifier=modifier, sub_one=sub_one)
                error_percent_offset.append(off_err)
                new_offset.append(offset)
                slope, sl_err = add_error(row[4], tens=tens, modifier=modifier, sub_one=sub_one)
                error_percent_slope.append(sl_err)
                new_slope.append(slope)
            else:
                target.append(1)
                error_percent_slope.append(0)
                error_percent_offset.append(0)
                new_slope.append(row[4])
                new_offset.append(row.MassOffset)
        
        
    err_data['target'] = target
    err_data['err_prop_slope'] = error_percent_slope
    err_data['err_prop_offset'] = error_percent_offset
    err_data['Mass/Time'] = new_slope
    err_data['MassOffset'] = new_offset
    return err_data


def mass_formula(channel, spec_bin_size, start_time,  mass_over_time, mass_offset):
    '''
    Apply formula for calculating mass at a channel.
    '''
    return ((channel * .001 * spec_bin_size + start_time) * mass_over_time + mass_offset)**2


def generate_calibrated_data(data):
    '''
    Applies mass_formula to every row in dataset to allow
    calibrated graphs to be generated.
    '''
    new_data = data.copy()
    masses = []
    channels = []
    intensities = []
    for row in new_data.itertuples():
        mass = []
        channel = []
        intensity = []
        spec = row.SpecBinSize
        m_over_t = row[4]
        m_offset = row.MassOffset
        time = row.StartFlightTime
        for tup in row.peaks:
            mass.append(mass_formula(tup[0], spec, time, m_over_t, m_offset))
            channel.append(tup[0])
            intensity.append(tup[1])
        intensities.append(intensity)   
        masses.append(mass)
        channels.append(channel)
    new_data['mass_channels'] = channels
    new_data['masses'] = masses
    new_data['intensities'] = intensities
    return new_data


def get_peaks(channels, height=50):
    '''
    Takes peaks for spectrum and returns all peaks and
    channel locations above given height, default height is 100.
    '''
    peak_loc, heights = find_peaks(channels, height=height)
    return list(zip(peak_loc, heights['peak_heights'].astype(int)))


def get_best_peaks(nbest, peaks):
    '''
    Return list of the n best or most spread out peaks for each data point.
    '''
    results = []
    for tups in peaks:
        best_peaks = []
        def loc(a):
            return a[0]
        
        tups.sort(key=loc)
        tups = np.array(tups)
        best_peaks.append(tups[0])
        best_peaks.append(tups[-1])
        indices = np.random.choice(np.array(range(len(tups[1:-1]))), nbest - 2)
        for index in indices:
            best_peaks.append(tups[index])
    
        results.append(best_peaks)
    return results


def get_min_peaks(err_data):
    '''
    Determine the smallest number of peaks the database has.
    '''
    min_peaks = len(err_data['channel, peak'][0])
    for tups in err_data['channel, peak']:
        if min_peaks > len(tups) :
            min_peaks = len(tups)
    return min_peaks


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

    
def get_pred_data(preds, data, names = ['xgb', 'lgbm', 'rfc']):
    '''
    Using output of get_wrong_preds, gets dataframe representing how
    models performed on the examples which were incorrectly classified.
    '''
    def loc(tup):
        return tup[0]
    pred_data = data.copy()
    for num in range(len(names)):
        preds[num].sort(key=loc)
        pred_data[names[num]] = np.array(preds[num])[:, 1]
    return pred_data


def get_dist_from_int(num):
    '''
    Returns distance from nearest whole number.
    '''
    return abs(num - round(num))


def get_avg(series, func=get_dist_from_int, args=None):
    '''
    Returns avg value of a function for every value in a series. Default function is 
    get_dist_whole_num.
    '''
    avg_dists = []
    for peaks in series:
        avg_dist = 0
        for i, peak in enumerate(peaks):
            dist = 0
            if args:
                dist = func(peak, *args)
            else:
                dist=func(peak)
            
            avg_dist += dist
        avg_dists.append(avg_dist / len(peaks))
    return avg_dists


def get_peaks_near_nom_masses(masses, nom_masses, num, rev=False):
    '''
    For a list of masses and a list of isotope masses near nominal
    masses, generate list of suspicious peaks far from nominal
    mass. Returns first 20 suspicious peaks.
    '''
    dists = []
    for mass in masses:
        index = round(mass)
        dists.append(mass - nom_masses[index])
    peaks_near = [x for _,x in sorted(zip(dists,masses), key=lambda pair: pair[0])]
    dists = sorted(dists)
    if rev:
        peaks_near.reverse()
        dists.reverse()
    return peaks_near[0:num], dists[0:num]


def get_avg_p_beyond(num_peaks_beyond, nom_masses, ab=True):
    '''
    Returns the avg number of peaks beyond isotope mass.
    If ab, uses absolute value.
    '''
    avg = 0
    for peak in num_peaks_beyond:
        if ab:
            avg += abs(peak - nom_masses[round(peak)] )/ len(num_peaks_beyond)
        else:
            avg += peak - nom_masses[round(peak)] / len(num_peaks_beyond)
    return avg


def get_peaks_below_nom_masses(masses, nom_masses):
    '''
    Returns all peaks in masses which are below lowest isotope mass 
    contained in nom_masses.
    '''
    peaks_below = []
    for mass in masses:
        index = round(mass)
        if nom_masses[index] > mass:
            peaks_below.append(mass)
    return peaks_below


def get_dist_nom_mass(peak, nom_masses):
    '''
    Return distance from closest lowest isotope mass.
    '''
    return peak - nom_masses[round(peak)]


def get_error_masses(dataframe, avgs_only=True,
 func=get_dist_from_int, args=None, add_to='both'):
    '''
    Adds error to every row in data frame, returns avg value of a function 
    for each modified row. Default function is get_dist_from_int.
    '''
    mass_lists = []
    slopes = []
    offsets = []
    slope_errs = []
    off_errs = []
    for row in dataframe.itertuples():
        slope, slope_err = None, None
        offset, off_err = None, None
        if add_to == 'both':
            slope, slope_err = add_error(row[4], 2, 3, 1)
            offset, off_err = add_error(row.MassOffset, 2, 3, 1)
        elif add_to == 'slope':
            slope, slope_err = add_error(row[4], 2, 3, 1)
            offset, off_err = row.MassOffset, 0.0
        else:
            slope, slope_err = row[4], 0.0
            offset, off_err = add_error(row.MassOffset, 2, 3, 1)
        slopes.append(slope)
        offsets.append(offset)
        slope_errs.append(slope_err)
        off_errs.append(off_err)
        spec = row.SpecBinSize
        time = row.StartFlightTime
        masses = []
        for tup in row.peaks:
            masses.append(mass_formula(tup[0], spec, time, slope, offset))
        mass_lists.append(masses)
    avgs = get_avg(mass_lists, func=func, args=args)
    if avgs_only:
        return avgs
    return mass_lists, avgs, off_errs, slope_errs


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

def get_isotope_mass_list(df, high, num=500):
    '''
    Return list of lowest or highest isotope masses in given dataframe at
    each nominal mass. If high returns highest masses, if low, the lowest.
    '''
    iso_masses = [num for num in range(num)]
    for masses in df['Isotope Masses']:
        for mass in masses:
            boolean = None
            if not high:
                boolean = mass < iso_masses[round(mass)]
            else: 
                boolean =  mass > iso_masses[round(mass)]

            if isinstance(iso_masses[round(mass)], int) or boolean:
                iso_masses[round(mass)] = mass
    return iso_masses


def get_peak_data(df, nom_masses_low, nom_masses_high, num=20, threshold=0.1, masses='masses', prefix=''):
    '''
    Takes in calibrated ToF DataFrame with peak masses, gets suspicious peaks 
    above and below isotope masses. Returns new dataframe with a column for 
    peaks below and above, dists below and above, as well as avgs for both and
    number of peaks above and below.
    
    Arguments-----
    df: dataframe with mass and peak data
    num: how many suspicious peaks to store in dataframe, default gets 20 peaks
    nom_masses_low: list of lowest isotope mass at each nominal mass, ceiling value
    nom_masses_high: list of highest isotope mass at each nominal mass, floor value
    threshold: distance from cieling or floor needed to be considered an important peak
    '''
    data = df.copy()
    peak_data_below = data[masses].apply(get_peaks_near_nom_masses, args=(nom_masses_low, -1, False))
    peak_data_above = data[masses].apply(get_peaks_near_nom_masses, args=(nom_masses_high, -1, True))
    peaks_below = []
    peaks_above = []
    dists_below = []
    dists_above = []
    num_above = []
    num_below = []
    for i, tup in enumerate(peak_data_above):
        below = [x for x in peak_data_below[i][1] if x < -1 * threshold]
        above = [x for x in peak_data_above[i][1] if x >  threshold]
        num_above.append(len(above))
        num_below.append(len(below))
        zeroes_below = [0 for x in range(num - len(below))]
        zeroes_above = [0 for x in range(num - len(above))]
        _ = peak_data_below[i][0][0:max(len(below), num)]
        peaks_below.append((_ + zeroes_below)[0:num])
        _ = peak_data_above[i][0][0:max(len(above), num)]
        peaks_above.append((_ + zeroes_above)[0:num])
        dists_below.append((below + zeroes_below)[0:num])
        dists_above.append((above + zeroes_above)[0:num])

    data[prefix + 'peaks_below'] = peaks_below
    data[prefix + 'peaks_above'] = peaks_above
    data[prefix + 'dists_above'] = dists_above
    data[prefix + 'dists_below'] = dists_below
    data[prefix + 'avg_dist_below'] = data[prefix + 'peaks_below'].apply(get_avg_p_beyond, args=(nom_masses_low, True,))
    data[prefix + 'avg_dist_above'] = data[prefix + 'peaks_above'].apply(get_avg_p_beyond, args=(nom_masses_high, True,))
    data[prefix + 'num_peaks_below'] = num_above
    data[prefix + 'num_peaks_above'] = num_below
    return data


def dimen_reduc_tsne(data, prefix, num=20, comps=2, random_state=42):
    '''
    Takes dataframe created by get_peak_data and reduces the dimensions of the dists or peaks
    above or below isotope mass using tSNE.
    Arguments ------
    data: dataframe from get_peak_data
    prefix: either 'dists' or 'peaks'
    num: min number of peaks to look at, default 20
    comps: # of components to reduce to, default 2
    random_state: random state of tsne object
    '''
    ab_col = '' + prefix + '_above'
    bel_col = '' + prefix + '_below'
    tsne = TSNE(n_components=comps, random_state=random_state)
    above = np.vstack([np.array(x) for x in data[data[ab_col].apply(len) >=num][ab_col]])
    above = tsne.fit_transform(above)
    below = np.vstack([np.array(x) for x in data[data[bel_col].apply(len) >=num][bel_col]])
    below = tsne.fit_transform(below)
    return above, below


def dimen_reduc_pca(data, prefix, num=20, comps=2, random_state=42):
    '''
    Takes dataframe created by get_peak_data and reduces the dimensions of the dists or peaks
    above or below isotope mass using tSNE.
    Arguments ------
    data: dataframe from get_peak_data
    prefix: either 'dists' or 'peaks'
    num: min number of peaks to look at, default 20
    comps: # of components to reduce to, default 2
    random_state: random state of tsne object
    '''
    ab_col = '' + prefix + '_above'
    bel_col = '' + prefix + '_below'
    pca = PCA(n_components=comps, random_state=random_state)
    above = np.vstack([np.array(x) for x in data[data[ab_col].apply(len) >=num][ab_col]])
    above = pca.fit_transform(above)
    below = np.vstack([np.array(x) for x in data[data[bel_col].apply(len) >=num][bel_col]])
    below = pca.fit_transform(below)
    return above, below


def augment_spectra(data: pd.DataFrame, column: str, sign_offset=1,
              sign_slope=1, err=0.1):
    '''
    Method for adding error to offset, slope, or both in order to 
    see how this change affects a spectras change in position relative
    to the population mean. Returns list of new peaks calculated with
    changed slope and offset values.
    Arguments -------
    df: ToF data in dataframe with columns necessary to calculate mass 
    (i.e channels, offset, slope, flight time, bin size)
    column: string referring to which column to add error to
    can be 'offset', 'slope', 'both'
    sign_offset: sign of error to add to offset, default 1
    sign_slope: sign of error to add to slope, default 1
    err: proportion of value to add as error, default 0.1
    '''
    augmented_rows = []
    for row in data.itertuples():
        slope = row[4]
        offset = row.MassOffset
        if column == 'slope':
            slope = augment_value(row[4], err, sign=sign_slope)
        elif column == 'offset':
            offset = augment_value(row.MassOffset, err, sign=sign_offset)
        elif column == 'both':
            slope = augment_value(row[4], err, sign=sign_slope)
            offset = augment_value(row.MassOffset, err, sign=sign_offset)
        else:
            raise ValueError("Invalid column value, must be in 'offset'," + 
                             "'slope','both'")
        augmented_row = []
        spec = row.SpecBinSize
        time = row.StartFlightTime
        for tup in row.peaks:
            augmented_row.append(mass_formula(tup[0], spec, time, slope, offset))
        augmented_rows.append(augmented_row)
    return augmented_rows


def augment_value(value, amount=0.1, sign=-1):
    '''
    Takes in value and adds / subtracts amount of itself from
    itself based on sign.
    '''
    return value + sign * amount * value


def get_hydrocarbs(max_num_carbs):
    carbon = 12
    hydrogen = 1.00782
    hydrocarbs = []
    for c in range(1, max_num_carbs):
        hydrocarbs.append(c * carbon + (2 * c + 1) * hydrogen)

    expanded_hydrocarbs = []
    for i, hydrocarb in enumerate(hydrocarbs):
        expanded_hydrocarbs.append(hydrocarb)
        if i < len(hydrocarbs) - 1:
            prev = hydrocarb
            diff = .01564
            dist = int(hydrocarbs[i+1]) - int(hydrocarb) - 1
            for num in range(dist):
                mass = prev + 1 + diff / dist
                expanded_hydrocarbs.append(mass)
                prev = mass
    return expanded_hydrocarbs


def get_ranges(mass_lists, length, max = 235):
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


def get_peak_suspiciousness(masses, ranges, show_correct_peaks=False, proportions=False, mass_thresh=2000):
    '''
    Returns list of how suspicious peaks are, how far into no mans land they are.
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
    susses = get_peak_suspiciousness(masses, ranges, False)
    a = np.array(masses)
    b = np.array(susses)
    return a[(b > thresh) & (a < 800)]


def get_xs(data, x=12, thresh=0.1):
    '''
    Get all peaks in data near a specific mass x.
    '''
    xs = []
    for row in data.itertuples():
        row_x = -1
        max = -1
        for i, mass in enumerate(row.masses):
            dif = abs(mass-x)
            inten = row.intensities[i]
            if dif < thresh and (inten > max or max == -1):
                max = inten
                row_x = dif
        xs.append(row_x)
    return xs


def get_frags(loc='../data/FragLibData32_Converted2010 (1)/Fragment Table.csv'):
    df = pd.read_csv(loc)
    a = list(df.columns[0:])
    a[0] = float(a[0])
    a[-1] = int(a[-1])
    a[-2] = a[-2][0]
    df.columns = ['FragmentMass', 'FragmentLabel', 'Isotopes', 'Formula', 'FragmentID']
    b = {'FragmentMass': a[0], 'FragmentLabel': a[1], 'Isotopes':a[2] , 'Formula':a[3],'FragmentID':a[4]}
    df = pd.concat([pd.DataFrame(b, index=[0]), df], sort=False)
    return df


