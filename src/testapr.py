"""
Main file for testing atomic_pattern_recognizer objects
"""
from atomic_pattern_recognizer import AtomicPatternRecognizer
from data_generation import read_csvs, generate_calibrated_data


def main():
    test = AtomicPatternRecognizer("../test_data/Elements.txt")
    test_data = read_csvs('../test_data/ThirtyTwoMoreFromGreg/')
    test_data = generate_calibrated_data(test_data)
    test_data['masses'] = test_data['masses'].apply(list)

    row = test_data.loc[10]
    elements = test.find_atomic_patterns(row['masses'], row['intensities'],
                                         thresh=.03)
    print(elements)


if __name__ == '__main__':
    main()