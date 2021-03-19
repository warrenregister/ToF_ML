from pandas import merge
from data_generator import DataGenerator
from data_transformation import get_better_spectra, get_precise_peaks

def data_setup():
    '''
    Performs perfunctory first time setup for data.
    '''
    dg = DataGenerator('../data/classification_cas_data.csv')
    norm_data = dg.df()
    data = get_better_spectra(dir='../data/SpectraCsvFiles_BkgndSubtractWatsonPeakFinder/')
    norm_data.sort_values('file_name', inplace=True)
    data.sort_values('file_name', inplace=True)
    norm_data = merge(data, norm_data, on='file_name')
    peaks = get_precise_peaks(norm_data, ['precise_channels', 'precise_intensities'])
    norm_data['peaks'] = peaks
    dg.set_df(norm_data)
    return dg
