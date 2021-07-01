"""
Main file for testing IsotopeFinder objects
"""
from isotope_finder import IsotopeFinder
from data_generation import read_csvs, generate_calibrated_data


def main():
    test = IsotopeFinder("../test_data/Elements.txt")
    test_data = read_csvs('../test_data/ThirtyTwoMoreFromGreg/')
    test_data = generate_calibrated_data(test_data)
    test_data['masses'] = test_data['masses'].apply(list)

    row = test_data.loc[0]
    elements = test.find_atomic_patterns(row['masses'], row['intensities'],
                                         thresh=.03)
    print(elements)


if __name__ == '__main__':
    main()