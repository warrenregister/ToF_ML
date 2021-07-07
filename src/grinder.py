"""
Contains grinder object for automatically fixing spectra

Steps:

Get spectrum
Apply grid of changes to spectrum, store each modified spectrum in new dataframe
Calibrate Dataframe with data_generation.generate_calibrated_data
Get Features with extracting_features.calc_feature_df
Get predictions from tof_calibration_detector.TOFCalibrationDetector.predict
Choose best calibration
Repeat until improvement stops
"""
from isotope_finder import IsotopeFinder
from tof_calibration_detector import ToFCalibrationDetector


class Grinder:

    def __init__(self, model_path, element_path):
        """
        Load in objects associated with classification and feature generation.

        Arguments -------
        model_path: path to stored ToFCalibrationDetector
        element_path: path to table of elemental data
        """
        self._isotope_finder = IsotopeFinder(element_path)
        self._classifier = ToFCalibrationDetector(dire=model_path)