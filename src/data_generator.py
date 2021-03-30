'''
Class which makes working with spectra data a lot cleaner by
opening, loading, and doing simple modification to data.
'''

import pandas  as pd 
import numpy as np
from math import log
from process_cas import generate_csv
from data_transformation import generate_calibrated_data, generate_data
from data_transformation import get_peaks

class DataGenerator():
    '''
    Wrapper around dataframe which can apply changes and return them.
    '''
    def __init__(self, df):
        '''
        Initiate dataframe using csv files or by converting cas files.
        '''
        if isinstance(df, pd.DataFrame):
            self._data = df
        else:
            print('Error illegal df argument, must be file path or pandas.core.frame.DataFrame')
    
    def df(self):
        '''
        return default dataframe
        '''
        return self._data
    
    def error_df(self, low_proportion, high_proportion, use_ranges=False,
     ranges=[0.2, 0.4, .6], cat=False, slope_index=4):
        '''
        Adds error to Mass/Time and MassOffset columns, error ranges beteween
        low_proportion and high_proportion. Returns new dataframe.
        '''
        tens = log(1 / high_proportion, 10)
        modifier = 1 / (1 - 10**tens * low_proportion)
        return generate_data(self._data, tens=tens, modifier=modifier,
                             use_ranges=use_ranges, ranges=ranges,
                             slope_cat=cat, slope_index=slope_index)

    def calibrated_df(self, error=False, low_proportion=.005, high_proportion=.01,
     use_ranges=False, ranges=[0.2, 0.4, .6], cat=False, slope_index=4):
        '''
        Using channels, slope, offset, and start time calibrates each spectras
        data and adds intensity, mass, and channel rows to returned dataframe.
        '''
        df = self._data.copy()
        if error:
            df = self.error_df(low_proportion, high_proportion, use_ranges,
                               ranges, cat, slope_index)
        return generate_calibrated_data(df, slope_index=slope_index)
    
    def apply_func(self, column, func):
        '''
        Apply func to column of dataframe.
        '''
        self._data[column].apply(func)
    
    def set_df(self, new_df):
        '''
        Sets underlying dataframe to new_df
        '''
        self._data = new_df
