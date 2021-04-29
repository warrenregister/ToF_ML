'''
Class which makes working with spectra data a lot cleaner by
opening, loading, and doing simple modification to data.
'''

import pandas  as pd
from ast import literal_eval
from math import log
from data_generation import generate_calibrated_data, generate_data

class DataGenerator():
    '''
    Wrapper around dataframe which can apply changes and return them.
    '''
    def __init__(self, df):
        '''
        Initiate wrapper on an existing dataframe or by reading in a dataframe.
        '''
        if isinstance(df, pd.DataFrame):
            self._data = df
        elif isinstance(df, str):
            try:
                self._data = pd.read_csv(df)
                for key in self._data.columns:
                    if isinstance(self._data[key][0], str):
                        if self._data[key][0][0] == "[":
                            series = self._data[key].apply(literal_eval)
                            self._data[key] = series
            except FileNotFoundError as e:
                print("Error: Invalid file path passed.")

        else:
            print("Error: illegal df argument, must be file path or" +
                  "pandas.core.frame.DataFrame")
    
    def df(self) -> pd.DataFrame:
        '''
        return default dataframe
        '''
        return self._data
    
    def error_df(self, low_proportion, high_proportion, ranges=(0.2, 0.4, .6),
                 multi_class=False, slope_cat=False, sub_one=False,
                 bad_amt=.0035, good_amt=.0015, slope_mods=False,
                 slope_modifiers=(3, 2)) -> pd.DataFrame:
        '''
        Adds error to Mass/Time and MassOffset columns, error ranges between
        low_proportion and high_proportion. Returns new dataframe.

        Arguments -------
        high_proportion: highest amount of error added to spectra
        low_proportion: lowest amount of error added to spectra

        See generate_data arguments for the rest.
        '''
        tens = log(1 / high_proportion, 10)
        modifier = 1 / (1 - 10**tens * low_proportion)
        return generate_data(self._data, tens, modifier, ranges,multi_class,
                             slope_cat, sub_one,bad_amt, good_amt,
                             slope_mods,slope_modifiers)

    def calibrated_df(self, error=False, low_proportion=.005,
                      high_proportion=.01, ranges=(0.2, 0.4, .6),
                      multi_class=False, slope_cat=False, sub_one=False,
                      bad_amt=.0035, good_amt=.0015, slope_mods=False,
                      slope_modifiers=(3, 2)) -> pd.DataFrame:
        '''
        Using channels, slope, offset, and start time calibrates each spectras
        data and adds intensity, mass, and channel rows to returned dataframe.
        '''
        df = self._data.copy()
        if error:
            df = self.error_df(low_proportion, high_proportion, ranges,
                               multi_class, slope_cat, sub_one,bad_amt,
                               good_amt, slope_mods, slope_modifiers)
        return generate_calibrated_data(df)
    
    def apply_func(self, column, func) -> pd.DataFrame:
        '''
        Apply func to column of dataframe.
        '''
        if column not in list(self._data.columns):
            print('Error: column not in dataframe.')
        else:
            self._data[column].apply(func)
        return self._data
    
    def set_df(self, new_df:pd.DataFrame) -> pd.DataFrame:
        '''
        Sets underlying dataframe to new_df, returns the old dataframe.
        '''
        old = self._data
        self._data = new_df
        return old