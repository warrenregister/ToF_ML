"""
Main file for testing atomic_pattern_recognizer objects
"""
from data_generation import (read_csvs, generate_calibrated_data,
                             get_isotope_data)
from calculate_spectra_stats import calc_npz_stats, get_ranges


def main():
    test_data = read_csvs("C:\\Users\\warre\\Desktop\\ML_TestData_FromGregGio_2021-06-15\\")
    test_data = generate_calibrated_data(test_data)
    test_data['masses'] = test_data['masses'].apply(list)

    spectrum = test_data.loc[5]
    fragments = get_isotope_data("../test_data/Elements.txt")['Isotope Masses']
    ranges = get_ranges(fragments, mass_limit=717, negative=True)
    stats = calc_npz_stats(spectrum.masses, spectrum.intensities, ranges)
    print(stats)

if __name__ == '__main__':
    main()
