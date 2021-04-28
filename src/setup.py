from pandas import read_csv
from data_generator import DataGenerator
from ast import literal_eval


def data_setup(loc='../data/all_calibrated_data.csv'):
    '''
    Performs perfunctory first time setup for data stored in 
    all_calibrated_data.csv.
    '''
    norm_data = read_csv(loc)
    norm_data['channels']  = norm_data['precise_channels'].apply(literal_eval)
    norm_data['intensities'] = norm_data['precise_intensities'].apply(literal_eval)
    norm_data.drop(['precise_channels', 'precise_intensities'],
                   axis=1, inplace=True)
    norm_data.dropna(inplace=True)
    norm_data = norm_data[norm_data['intensities'].apply(len)> 0].copy()
    dg = DataGenerator(norm_data)
    return dg
