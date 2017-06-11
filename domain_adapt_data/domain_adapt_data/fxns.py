import numpy as np
import os
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import python_utils.python_utils.caching as caching

home_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = '%s/%s' % (home_folder, 'data')


def standard_format(n_rows, data_file, sep=',', delim_whitespace=False):
    if not delim_whitespace:
        df = pd.read_csv(data_file, header=None, sep=sep, index_col=None)
    else:
        df = pd.read_csv(data_file, header=None, delim_whitespace=True)
    xs = df.iloc[:n_rows,:-1].values
    ys = df.iloc[:n_rows,-1].values
    return xs, ys

def CA_housing(n_rows=-1):
    data_file = '%s/cal_housing.data' % data_folder
    return standard_format(n_rows, data_file)


def boston_housing(n_rows=-1):
    data_file = '%s/boston_housing.data' % data_folder
    return standard_format(n_rows, data_file, delim_whitespace=True)



def subsample_indicator(xs, f):
    raw = np.array(map(f, xs))
#    raw = raw / raw.sum()
    raw = raw / raw.max()
    return np.random.uniform(size=len(xs)) < raw


def kmm_paper_pca_subsample(a, b, sigma, xs, ys):
    # do pca
    from sklearn.decomposition import KernelPCA
    pca = KernelPCA(n_components=1, gamma=1./(2*(sigma**2)))
    pca.fit(xs)
    the_component = pca.transform(xs)[:,0]
    assert np.min(the_component) < 0
    if np.abs(np.min(the_component)) > np.max(the_component):
        the_component = -1. * the_component
    mean = np.min(the_component) + ((np.mean(the_component) - np.min(the_component)) / a)
    variance = (np.mean(the_component) - np.min(the_component)) / b
    f = lambda x: np.exp(-(x - mean)**2 / (2 * variance))
    keep = subsample_indicator(the_component, f)
    if True:
        fig,ax = plt.subplots()
        ax.hist(the_component, label='all', normed=True)
        ax.hist(the_component[keep], label='keep', normed=True)
        ax.set_title('kmm paper subsampling')
        ax.set_xlabel('first kernel PCA component')
        ax.legend()
        caching.fig_archiver.archive_fig(fig)
        #basic.display_fig_inline(fig)
        
    return xs[keep], ys[keep]


def doubly_robust_paper_pca_subsample(alpha, xs, ys):
    # do pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(xs)
    the_component = pca.transform(xs)[:,0]
    mean = np.min(the_component) + (alpha * (np.max(the_component) - np.min(the_component)))
    std = 0.5 * np.std(the_component)
    f = lambda x: np.exp(-(x - mean)**2 / (2 * (std**2)))
    keep = subsample_indicator(the_component, f)
    if True:
        alpha = 0.5
        fig,ax = plt.subplots()
        ax.hist(the_component, label='all', normed=True, alpha=alpha)
        ax.hist(the_component[keep], label='keep', normed=True, alpha=alpha)
        ax.set_title('doubly robust paper subsampling')
        ax.set_xlabel('first PCA component')
        ax.legend()
        caching.fig_archiver.archive_fig(fig)
        #basic.display_fig_inline(fig)
    return xs[keep], ys[keep]
    
        
def split(training_proportion, seed, xs, ys):
    training_indicator = np.random.uniform(size=len(xs)) < training_proportion
    return xs[training_indicator],ys[training_indicator],  xs[~training_indicator], ys[~training_indicator]

def get_data_helper(training_proportion, training_sampler, get_data_f, num_data, seed):

    # get_data
    np.random.seed(seed)
    xs, ys = get_data_f(num_data)
    original_xs_shape, original_ys_shape = xs.shape, ys.shape
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit_transform(xs)
    xs_train, ys_train, xs_test, ys_test = split(training_proportion, seed, xs, ys)
    original_xs_train_shape, original_ys_train_shape = xs_train.shape, ys_train.shape
    original_xs_test_shape, original_ys_test_shape = xs_test.shape, ys_test.shape
    xs_train, ys_train = training_sampler(xs_train, ys_train)
    sampled_xs_train_shape, sampled_ys_train_shape = xs_train.shape, ys_train.shape

    # print sizes
    size_info_list = [
        'seed: %d' % seed,
        'original_xs: %s, original_ys_shape: %s' % (original_xs_shape, original_ys_shape),
        'original_xs_train: %s original_ys_train: %s' % (original_xs_train_shape, original_ys_train_shape),
        'original_xs_test: %s original_ys_test: %s' % (original_xs_test_shape, original_ys_test_shape),
        'sampled_xs_train: %s sampled_ys_train: %s' % (sampled_xs_train_shape, sampled_ys_train_shape),
        ]
    caching.fig_archiver.fig_text(size_info_list)

    # print weights
    max_weight = 10.        
    import domain_adapt.domain_adapt.new.fxns
    kmm_ws_train = domain_adapt.domain_adapt.new.kmm.get_cvxopt_KMM_ws_sigma_median_distance(max_weight, 0.01, xs_train, xs_test)
    import domain_adapt.domain_adapt.new.constructors
    c_logreg = 1.
    logreg_ws_train = domain_adapt.domain_adapt.new.constructors.logreg_ratios(c_logreg, max_weight, xs_train, xs_test)
    fig, ax = plt.subplots()
    ax.scatter(kmm_ws_train, logreg_ws_train)
    ax.set_xlabel('kmm weights')
    ax.set_ylabel('logreg weights')
    ax.set_title('kmm_size: %.2f logreg_size: %.2f' % (len(xs_train)**2 / (np.linalg.norm(kmm_ws_train)**2), len(xs_train)**2 / (np.linalg.norm(logreg_ws_train)**2)))
    caching.fig_archiver.archive_fig(fig)
    #basic.display_fig_inline(fig)

    return xs_train, xs_test, ys_train, ys_test

"""
don't need below stuff yet
"""


class random_train_test_data_iterator(object):

    def __init__(self, training_proportion, training_sampler, get_data_f, num_data, num_splits):
        self.training_proportion, self.training_sampler, self.get_data_f, self.num_data, self.num_splits = training_proportion, training_sampler, get_data_f, num_data, num_splits

    def __iter__(self):
        for i in xrange(self.num_splits):
            yield get_data_helper(self.training_proportion, self.training_sampler, self.get_data_f, self.num_data, i)


class shift_Xy_data_iterator(object):

    def __init__(self, horse):
        self.horse = horse

    def __iter__(self):
        from domain_adapt.domain_adapt.kernels import matrices_to_shift_Xy
        for (xs_train, xs_test, ys_train, ys_test) in self.horse:
            yield matrices_to_shift_Xy(xs_train, ys_train, xs_test, ys_test)
            
