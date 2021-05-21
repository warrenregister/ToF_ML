"""
Methods for automatically calibrating a TOF Spectrum based on some core
statistics.
"""
from numpy import linspace, where
from pandas import DataFrame
from data_generation import get_frags, get_isotope_data
from data_generator import DataGenerator
from calculate_spectra_stats import calc_new_spectrum_stats, get_fragment_stats
from calculate_spectra_stats import get_ranges


def main():
    data_loc = "../test_data/unmodified_full_dataset.csv"
    fragment_loc = "../test_data/Fragment Table.csv"
    iso_loc = "../test_data/Elements.txt"

    dg = DataGenerator(data_loc)
    df = dg.calibrated_df()
    df = get_fragment_stats(df, prop_name='original_proportion_identified')

    df.reset_index(inplace=True)  # legacy functions rely on column order
    a = df['index']
    df.drop('index', axis=1, inplace=True)
    df['index'] = a

    npzs = get_ranges(get_isotope_data(iso_loc)['Isotope Masses'], 2000)

    offsets = []
    slopes = []
    files = []
    indices = []
    i = 0
    for row in df.loc[0:].itertuples():
        print('Spectrum Number: ' + str(i))
        p, o, s = get_best_offset(row, [.00001, .0000001], [.001, .000001],
                                  fragment_loc, npzs, offsets=20, slopes=10)
        print('Final Proportion: ' + str(p))
        offsets.append(o)
        slopes.append(s)
        files.append(row.file_name)
        indices.append(row.index)
        i += 1

    DataFrame({'offsets': offsets, 'slopes': slopes, 'names': files,
               'indices': indices}).to_csv(
        '../data/updated_calibration_new_1900_first_478.csv', index=False)


def get_best_offset(spectrum, slope_range, offset_range, loc, npzs, offsets=30,
                    slopes=20, prev=0, mnpzps=None, mnpzad=None) -> tuple:
    """
    Find best amount of slope/offset to add/subtract to achieve the optimal
    calibration for a spectrum. Calibration is measured using mass fragments.
    A spectrum with more matches to known frags is more calibrated than one
    with fewer. A spectrum whose matches are very close to known mass is more
    calibrated than one with some that are further away.

    Returns tuple containing the best performing offset and slope changes.


    Arguments -------
    spectrum: row from dataframe containing information on a spectrum
    slope_range: data structure containing min, max slope to try, slope errors
    are typically smaller than offset errors.
    offset_range: data structure containing min, max offset augmentation
    to try. This method shrinks the range iteratively until
    the best offset is achieved.
    loc: file location of fragment database
    npzs: no peak zone list, sometimes called ranges
    offsets: number of evenly spaced offsets to generate in range
    slopes: number of evenly spaced slopes to generate in range
    prev: float, for recursion: previous best proportion
    mnpzps: int, for recursion: max number of no peak zone peaks allowable
    mnpzad: float, for recursion: max avg distance into no peak zone allowable
    """
    frags = get_frags(loc)['FragmentMass']

    best_prop = 0
    if mnpzps is None and mnpzad is None:
        print('optimizing ' + spectrum.file_name)
        prop, _, _, mnpzps, mnpzad = calc_new_spectrum_stats(spectrum, frags, 0,
                                                             0, npzs)
        best_prop = prop
        print('initial proportion: ' + str(prop))

    edge = 0
    s_edge = 0
    best_offset = 0
    best_slope = 0
    best_ld = 1
    best_hd = 1
    mults = [1, -1]
    s_space = linspace(slope_range[0], slope_range[1], slopes)
    for i in range(len(s_space)):
        slope = s_space[i]
        for slope_mult in mults:
            slope_val = slope * slope_mult
            space = linspace(offset_range[0], offset_range[1], offsets)
            for j in range(len(space)):
                offset = space[j]
                for mult in mults:
                    improved = False
                    offset_val = mult * offset
                    values = calc_new_spectrum_stats(spectrum, frags,
                                                     slope_val, offset_val,
                                                     npzs)
                    prop, low, high, num_npz, ad_npz = values
                    if (prop > best_prop and num_npz <= mnpzps and
                       ad_npz <= mnpzps):
                        improved = True
                    elif (prop == best_prop and best_hd - high > 0 and
                          best_ld - low > 0):
                        improved = True

                    if improved:
                        best_prop = prop
                        best_offset = offset_val
                        best_slope = slope_val
                        best_ld = low
                        best_hd = high
                        edge = where(space == mult * best_offset)
                        edge = 2 * abs(.5 - (edge[0][0] + .1) / len(space))
                        s_edge = where(s_space == slope_mult * best_slope)
                        s_edge = abs(.5 - (s_edge[0][0] + .1) / len(s_space))
                        s_edge *= 2
    print(best_prop)
    if best_prop > prev:
        a = best_offset - (.5/edge) * best_offset
        b = best_offset + (.5/edge) * best_offset
        c = best_slope + (.5 / s_edge) * best_slope
        d = best_slope - (.5 / s_edge) * best_slope
        bp, bs, bo = get_best_offset(spectrum,
                                     slope_range=[c, d],
                                     offset_range=[a, b],
                                     prev=best_prop,
                                     offsets=20,
                                     loc=loc,
                                     npzs=npzs,
                                     slopes=5,
                                     mnpzps=mnpzps,
                                     mnpzad=mnpzad)
        if bp >= best_prop:
            best_prop = bp
            best_offset = bo
            best_slope = bs
    return best_prop, best_slope, best_offset


if __name__ == '__main__':
    main()
