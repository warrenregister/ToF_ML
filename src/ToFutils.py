import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from scipy.signal import find_peaks


def add_error(number, modifier=2, tens=2, threshold=0.5):
    '''
    Add or subtract a random amount of error based on the modifer and tens variables.
    Error calculated : (1 - random num between 0 and 1 / modifier) / ten**tens
    '''
    randoms = np.random.rand(2)
    error = (1 - randoms[0] / modifier) / 10**tens
    new_num = 0
    if randoms[1] < threshold:
        new_num  = number - number * error
        error = error * -1
    else:
        new_num = number + number * error
    return new_num, error


def generate_data(data, tens, modifier, use_ranges=False, ranges=[0.2, 0.4, .6]):
    '''
    Takes calibrated ToF-MS data and adds error to the offset and mass. If use_ranges returns
    multiclass classification dataset 0: no error, 1: offset error, 2: slope error, 3: both.

    Arguments -------
    data: dataframe to add error to
    tens: power of tens place for error
    modifier: modifier to apply to error, eg (1  - rand num) / modifier
    use_ranges: boolean if true returns multiclass dataset for combinations of error.
    ranges: percentile ranges for different outcomes slope error, offset
    and slope error, offset error
    '''
    err_data = data.copy()
    target = []
    error_percent_slope = []
    error_percent_offset = []
    new_slope = []
    new_offset = []
    for row in err_data.iterrows():
        num = np.random.rand(1)
        if use_ranges:
            if num < ranges[0]: #slope only
                target.append(2)
                slope, sl_err = add_error(row[1]['Mass/Time'], tens=tens, modifier=modifier)
                error_percent_slope.append(sl_err)
                new_slope.append(slope)
                new_offset.append(row[1]['MassOffset'])
                error_percent_offset.append(0)
            elif num >= ranges[0] and num < ranges[1]: # both
                target.append(3)
                offset, off_err = add_error(row[1]['MassOffset'], tens=tens, modifier=modifier)
                error_percent_offset.append(off_err)
                new_offset.append(offset)
                slope, sl_err = add_error(row[1]['Mass/Time'], tens=tens, modifier=modifier)
                error_percent_slope.append(sl_err)
                new_slope.append(slope)
            elif num >= ranges[1] and num < ranges[2]: # offset only
                target.append(1)
                error_percent_slope.append(0)
                offset, off_err = add_error(row[1]['MassOffset'], tens=tens, modifier=modifier)
                error_percent_offset.append(off_err)
                new_offset.append(offset)
                new_slope.append(row[1]['Mass/Time'])
            else:
                target.append(0)
                error_percent_slope.append(0)
                error_percent_offset.append(0)
                new_slope.append(row[1]['Mass/Time'])
                new_offset.append(row[1]['MassOffset'])
        else:
            if num < 0.5:
                target.append(0)
                offset, off_err = add_error(row[1]['MassOffset'], tens=tens, modifier=modifier)
                error_percent_offset.append(off_err)
                new_offset.append(offset)
                slope, sl_err = add_error(row[1]['Mass/Time'], tens=tens, modifier=modifier)
                error_percent_slope.append(sl_err)
                new_slope.append(slope)
            else:
                target.append(1)
                error_percent_slope.append(0)
                error_percent_offset.append(0)
                new_slope.append(row[1]['Mass/Time'])
                new_offset.append(row[1]['MassOffset'])
        
        
    err_data['target'] = target
    err_data['err_prop_slope'] = error_percent_slope
    err_data['err_prop_offset'] = error_percent_offset
    err_data['Mass/Time'] = new_slope
    err_data['MassOffset'] = new_offset
    return err_data


def mass_formula(channel, spec_bin_size,start_time,  mass_over_time, mass_offset):
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
    for row in new_data.iterrows():
        mass = []
        channel = []
        intensity = []
        spec = row[1]['SpecBinSize']
        m_over_t = row[1]['Mass/Time']
        m_offset = row[1]['MassOffset']
        time = row[1]['StartFlightTime']
        for tup in row[1]['peaks']:
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



def get_kfold_stats(X, y, nsplits, seed=33, models=None):
    '''
    Train models on X, y using kfold cross validation with nsplits. Models 
    defaults to XGBoost, LightGB, and RandomForestClassifier.
    Returns average accuracy of each model and the incorrect predictions
    and indexes of each incorrectly predicted point.

    Arguments -----
    X features / training variables
    y target / training labels
    nsplits (optional) default: 5 number of splits to use in kfold algorithm
    models (optional) list of ml models to train
    '''
    if not models:
        models = [XGBClassifier(), LGBMClassifier(), RandomForestClassifier(n_estimators=100, max_depth=None)]
    xlr_accs = [0 for model in models]
    xlr_index_pred = [[] for model in models]
    avg_feature_importance = [np.zeros(X.shape[1]) for model in models]
    for X_train, y_train, X_test, y_test, train_index, test_index in kfold(X, y, nsplits, seed=seed):
        for model in models:
            model.fit(X_train, y_train)
            
        for i, model in enumerate(models):
            acc, preds = model_acc(model, X_test, y_test)
            xlr_accs[i] += acc / nsplits
            xlr_index_pred[i] += zip(test_index, preds)
        avg_feature_importance[i] += model.feature_importances_
            
    return xlr_accs, xlr_index_pred, [x / nsplits for x in avg_feature_importance]


def kfold(X, y, k=5, stratify=False, shuffle=False, seed=33):
    """K-Folds cross validation iterator.

    Parameters
    ----------
    k : int, default 5
    stratify : bool, default False
    shuffle : bool, default True
    seed : int, default 33

    Yields
    -------
    X_train, y_train, X_test, y_test, train_index, test_index
    """
    if stratify:
        kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=shuffle)
    else:
        kf = KFold(n_splits=k, random_state=seed, shuffle=shuffle)
    
    data = np.array(X)
    target = np.array(y)
    for train_index, test_index in kf.split(X, y):
        X_train, y_train = data[train_index], target[train_index]
        X_test, y_test = data[test_index], target[test_index]
        yield X_train, y_train, X_test, y_test, train_index, test_index


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



def get_df(fold, model, err_data):
    new_df = err_data.loc[fold[0]]
    new_df[model] = fold[1]
    return new_df

    
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


def test_lgbm(lrs, n_leav, n_ests, m_depths, boost_type, seed_num):
    '''
    Test accuracy of lgbm for each value in lists per each perameter.
    '''
    accs = []
    params = []
    for num in n_leav:
        for lr in lrs:
            for n_est in n_ests:
                for depth in m_depths:
                    for boost in boost_type:
                        seed_acc = 0
                        for seed in np.random.randint(1, 100, size=seed_num):
                            model = LGBMClassifier(boosting_type=boost,max_depth=depth,
                                                   num_leaves=num, learning_rate=lr,
                                                   n_estimators=n_est)
                            acc, _, p = get_kfold_stats(X, y, 5, seed_num, models=[model])
                            seed_acc += acc[0] / 15
                        accs.append(seed_acc)
                        params.append([boost, lr, num, n_est, depth,])
    best_acc = max(accs)
    best_params = params[accs.index(max(accs))]
    return (accs, params, best_acc, best_params)


def get_distance_from_int(num):
    '''
    Returns distance from nearest whole number.
    '''
    return abs(num - round(num))


def model_acc(model, X_test, y_test):
    '''
    Returns accuracy and predictions of a model.
    '''
    preds = model.predict(X_test)
    return(accuracy_score(y_test, preds), preds)

def parameter_generator(parameters, names):
    '''
    Given a list of lists containing parameters, and a list of names
    yields every combination of the parameters in the lists.
    '''
    indices = [0 for x in names]
    max_vals = [len(x) for x in parameters]
    while indices[0] < len(parameters[0]):
        params = {}
        for i, param in enumerate(parameters):
            params[names[i]] = param[indices[i]]
        indices = increment_index(indices, max_vals)
        yield params
            
def increment_index(indices, max_vals):
    '''
    Recursively increments the indices of several lists so that
    every combination of elements of those lists can be seen.
    
    Arguments -------
    indices = list of indices for lists
    max_vals = length of each list
    '''
    indices[-1] += 1
    if indices[-1] > max_vals[-1] - 1 and len(indices) > 1:
        indices[-1] = 0
        indices[0:-1] = increment_index(indices[0:-1], max_vals[0:-1])
    return indices

def get_precise_peaks(df):
    all_peaks = []
    for row in df.iterrows():
        chans = row[1]['precise_channels'][row[1]['precise_intensities'] > 0]
        intens = row[1]['precise_intensities'][row[1]['precise_intensities'] > 0]
        all_peaks.append(list(zip(chans, intens)))
    return all_peaks

def get_better_spectra():
    dir = './data/SpectraCsvFiles/'
    channels = []
    intensities = []
    names = []
    for csv in os.listdir(dir):
        data = pd.read_csv(dir + csv, names=['channels', 'intensities'])
        channels.append(data['channels'])
        intensities.append(data['intensities'])
        names.append(csv[0:-3] + 'cas')
    return channels, intensities, names


def get_dist_whole_num(peak):
    return abs(peak - round(peak))


def get_avg(series, func=get_dist_whole_num, args=None):
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


def augment_value(value, amount=0.1, sign=-1):
    '''
    Takes in value and adds / subtracts amount of itself from
    itself based on sign.
    '''
    return value  + sign * amount * value


def get_peaks_near_nom_masses(masses, nom_masses):
    dists = []
    for mass in masses:
        index = round(mass)
        dists.append(mass - nom_masses[index])
    peaks_near = [x for _,x in sorted(zip(dists,masses))]
    return peaks_near[0:20], sorted(dists)[0:20]


def get_avg_p_below(n_p_below, ab=True):
    avg = 0
    for peak in n_p_below:
        if ab:
            avg += abs(peak - round(peak)) / len(n_p_below)
        else:
            avg += peak - round(peak) / len(n_p_below)
    return avg


def get_peaks_below_nom_masses(masses, nom_masses):
    peaks_below = []
    for mass in masses:
        index = round(mass)
        if nom_masses[index] > mass:
            peaks_below.append(mass)
    return peaks_below


def get_dist_nom_mass(peak, nom_masses):
    return peak - nom_masses[round(peak)]


def get_dist_whole_num(peak):
    return abs(peak - round(peak))


def plot_error_hists(low, high, pos=-1, func=get_dist_whole_num, args=None):
    df = dg.calibrated_df(error=True, low_proportion=low, high_proportion=high)
    df['avg_dist'] = get_avg(df['masses'], func, args)
    if pos==1:
        df = df[df['avg_dist'] >=0]
    elif pos==0:
        df = df[df['avg_dist'] <0]
    plt.hist(df['avg_dist'][df['target']==1], bins=50, alpha=0.5, label='Calibrated')
    plt.hist(df['avg_dist'][df['target']==0], bins=50, alpha=0.5, label='Error')
    plt.legend(loc='upper right')
    plt.show()
    return df


def get_error_masses(dataframe, avgs_only=True, func=get_dist_whole_num, args=None):
    mass_lists = []
    slopes = []
    offsets = []
    slope_errs = []
    off_errs = []
    for row in dataframe.iterrows():
        slope, slope_err = add_error(row[1]['Mass/Time'], 2, 2, 1)
        offset, off_err = add_error(row[1]['MassOffset'], 2, 2, 1)
        slopes.append(slope)
        offsets.append(offset)
        slope_errs.append(slope_err)
        off_errs.append(off_err)
        spec = row[1]['SpecBinSize']
        time = row[1]['StartFlightTime']
        masses = []
        for tup in row[1]['peaks']:
            masses.append(mass_formula(tup[0], spec, time, slope, offset))
        mass_lists.append(masses)
    avgs = get_avg(mass_lists, func=func, args=args)
    if avgs_only:
        return avgs
    return mass_lists, avgs, off_errs, slope_errs