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
    def __init__(self, df, cas_files=False, new_path=''):
        '''
        Initiate dataframe using csv files or by converting cas files.
        '''
        if type(df) == str :
            if cas_files:
                generate_csv(df)
                df = new_path
            data = pd.read_csv(df)
            try:
                data.drop('Unnamed: 0', axis=1, inplace=True)
            except:
                pass
            data.drop('sequence', inplace=True,axis=1)
            def str_to_list(string):
                return [int(s) for s in string[1:-1].split(',')]

            data['channels'] = data['channels'].apply(str_to_list)
            self._data = data
        elif isinstance(df, pd.DataFrame):
            self._data = df
        else:
            print('Error illegal df argument, must be file path or pandas.core.frame.DataFrame')
    
    def df(self):
        '''
        return default dataframe
        '''
        return self._data
    
    def error_df(self, low_proportion, high_proportion, use_ranges=False,
     ranges=[0.2, 0.4, .6], cat=False):
        '''
        Adds error to Mass/Time and MassOffset columns, error ranges beteween
        low_proportion and high_proportion. Returns new dataframe.
        '''
        tens = log(1 / high_proportion, 10)
        modifier = 1 / (1 - 10**tens * low_proportion)
        return generate_data(self._data, tens=tens, modifier=modifier, use_ranges=use_ranges, ranges=ranges, slope_cat=cat)

    def calibrated_df(self, error=False, low_proportion=.005, high_proportion=.01,
     use_ranges=False, ranges=[0.2, 0.4, .6], cat=False):
        '''
        Using channels, slope, offset, and start time calibrates each spectras
        data and adds intensity, mass, and channel rows to returned dataframe.
        '''
        df = self._data.copy()
        if error:
            df = self.error_df(low_proportion, high_proportion, use_ranges, ranges, cat)
        return generate_calibrated_data(df)
    
    def get_peak_data(self):
        '''
        Returns 2 dim list of nbest peaks per each spectra.
        '''
        peak_list = self._data['channels'].apply(get_peaks)
        return peak_list
    
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
