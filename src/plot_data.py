'''
Contains specific plotting functions
'''
from data_generator import DataGenerator
from data_transformation import get_avg, dimen_reduc_pca, dimen_reduc_tsne, get_peak_data
from data_transformation import get_avg_p_beyond, get_peaks_below_nom_masses, get_dist_from_int
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_hists(dg, low, high, pos=-1, func=get_dist_from_int, args=None):
    '''
    Takes in data generator object, creates new dataset with error between low and high proportions, 
    plots the avg value of a func over all rows in new data frame, returns new dataframe.
    '''
    df = dg.calibrated_df(error=True, low_proportion=low, high_proportion=high)
    df['avg_dist'] = get_avg(df['masses'], func, args)
    if pos==1:
        df = df[df['avg_dist'] >=0]
    elif pos==0:
        df = df[df['avg_dist'] <0]
    plt.hist(df['avg_dist'][df['target']==1], bins=50, alpha=0.5, label='Calibrated')
    plt.hist(df['avg_dist'][df['target']==0], bins=50, alpha=0.5, label='Error')
    plt.legend(loc='upper right')
    plt.show()
    return df


def plot_avg_num(df, low, high, nom_masses):
    generator = DataGenerator(df=df)
    generator.set_df(generator.error_df(low, high, use_ranges=True))
    data = generator.calibrated_df()
    data['peaks_below'] = data['masses'].apply(get_peaks_below_nom_masses, args=(nom_masses,))
    data['num_peaks_below'] = data['peaks_below'].apply(len)
    norm_avg_num = data['num_peaks_below'][data['target']==0]
    offset_avg_num = data['num_peaks_below'][data['target']==1]
    slope_avg_num = data['num_peaks_below'][data['target']==2]
    both_avg_num = data['num_peaks_below'][data['target']==3]
    plt.figure(figsize=(20, 10))
    plt.hist(norm_avg_num, bins=50, alpha=0.5, label='Calibrated')
    plt.hist(offset_avg_num, bins=50, alpha=0.5, label='Offset')
    plt.hist(slope_avg_num, bins=50, alpha=0.5, label='Slope')
    plt.hist(both_avg_num, bins=50, alpha=0.5, label='Both')
    plt.legend(loc='upper right')
    plt.title('Avg Number of Peaks Below Floor '  + str(high * 100) + '% - ' + str(low * 100) + '%')
    plt.xlabel('Number of Peaks')
    plt.ylabel('Count')
    plt.show()


def plot_avg_dist(df, low, high, nom_masses, lim=1):
    generator = DataGenerator(df=df)
    generator.set_df(generator.error_df(low, high, use_ranges=True))
    data = generator.calibrated_df()
    data['peaks_below'] = data['masses'].apply(get_peaks_below_nom_masses, args=(nom_masses,))
    data['avg_p_below_dist'] = data['peaks_below'].apply(get_avg_p_beyond, args=(nom_masses, True,))
    norm_avg_dist = data['avg_p_below_dist'][data['target']==0]
    offset_err_avg_dist = data['avg_p_below_dist'][data['target']==1]
    slope_err_avg_dist = data['avg_p_below_dist'][data['target']==2]
    both_err_avg_dist = data['avg_p_below_dist'][data['target']==3]
    plt.figure(figsize=(20, 10))
    plt.hist(norm_avg_dist, bins=50, alpha=0.5, label='Calibrated')
    plt.hist(offset_err_avg_dist, bins=50, alpha=0.5, label='Offset')
    plt.hist(slope_err_avg_dist, bins=50, alpha=0.5, label='Slope')
    plt.hist(both_err_avg_dist, bins=50, alpha=0.5, label='Both')
    plt.legend(loc='upper right')
    plt.title('Avg Distance From Floor ' + str(high * 100) + '% - ' + str(low * 100) + '%')
    plt.xlabel('Distances (amu)')
    plt.ylabel('Count')
    plt.xlim(0, lim)
    plt.show()


def plot_components(data, prefix='dists', tsne=True, num=20, state=42):
    '''
    Takes in data from get_peak_data and plots 2 components of it using tSNE
    or PCA for dimension reduction based on the value of tsne.
    '''
    data = data.copy()
    above, below = None, None
    if tsne:
        above, below = dimen_reduc_tsne(data, prefix, num, random_state=state)
    else:
        above, below = dimen_reduc_pca(data, prefix, num, random_state=state)
        
    data['' + prefix + '_below_component_x'] = below[:, 0]
    data['' + prefix + '_below_component_y'] = below[:, 1]
    data['' + prefix + '_above_component_x'] = above[:, 0]
    data['' + prefix + '_above_component_y'] = above[:, 1]
    fig, axs = plt.subplots(2)
    fig.set_size_inches(20, 10)
    hue = None
    if 'target' in data.columns:
        hue='target'
    sns.scatterplot('' + prefix + '_above_component_x', '' + prefix + '_above_component_y', data=data, style='technique', hue=hue, ax=axs[0], palette='Accent')
    sns.scatterplot('' + prefix + '_below_component_x', '' + prefix + '_below_component_y', data=data, style='technique', hue=hue, ax=axs[1], palette='Accent')
    plt.show()
    return data