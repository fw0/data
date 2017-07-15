import numpy as np
import os
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import python_utils.python_utils.caching as caching

home_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = '%s/%s' % (home_folder, 'data')


def standard_format(data_file, sep=',', delim_whitespace=False, n_rows=None):
    if not delim_whitespace:
        df = pd.read_csv(data_file, header=None, sep=sep, index_col=None)
    else:
        df = pd.read_csv(data_file, header=None, delim_whitespace=True)
    relevant_df = df.iloc[:n_rows,:]
    relevant_df = (relevant_df - relevant_df.mean()) / relevant_df.std()
    xs = np.concatenate((relevant_df.iloc[:,:-1].values, np.ones(shape=(len(relevant_df),1))), axis=1)
    ys = relevant_df.iloc[:,-1].values
    return xs, ys

def CA_housing(n_rows=None):
    data_file = '%s/cal_housing.data' % data_folder
    return standard_format(n_rows, data_file)


def boston_housing(n_rows=None):
    data_file = '%s/boston_housing.data' % data_folder
    return standard_format(n_rows, data_file, delim_whitespace=True)


def kin_data(name, n_rows=None):
    data_file = '%s/%s.data' % (data_folder, name)
    print data_file
    return standard_format(data_file, sep=',', delim_whitespace=True, n_rows=n_rows)

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


def lsif_data_subsample(xs, ys, N_train, N_test, seed=None, c_cols=None, same_dim=0, diff_dim=0):
    # assumes last column of xs is indicator feature
    if not (seed is None):
        np.random.seed(seed)
    if c_cols is None:
        c_cols = range(xs.shape[1])
    c = np.random.choice(c_cols)
    xs = xs - np.min(xs, axis=0)[np.newaxis,:]
    xs = xs / (np.append(np.max(xs[:,0:-1], axis=0), 1))[np.newaxis,:]
    xs = np.random.permutation(xs)
    xs[:,-1] = 1.
    p = np.minimum(1, 4*(xs[:,c]**2))
    keep = np.random.uniform(size=len(xs)) < p
    until = np.argmax(np.cumsum(keep.astype(int)) == N_test) + 1
    add_sd = 1.
    same_mean = 0.
    diff_mean = 1.
    xs_test = xs[0:until][keep[0:until]]
    xs_test = np.concatenate((xs_test, np.random.normal(same_mean,add_sd,size=(N_test,same_dim)), np.random.normal(diff_mean,add_sd,size=(N_est,same_dim))), axis=1)
    ys_test = ys[0:until][keep[0:until]]
    xs_train = xs[until:until+N_train]
    xs_train = np.concatenate((xs_train, np.random.normal(same_mean,add_sd,size=(N_train,same_dim)), np.random.normal(same_mean,add_sd,size=(N_train,same_dim))), axis=1)
    ys_train = ys[until:until+N_train]
#    print ys_test[0:10]
    assert len(xs_train) == N_train
    assert len(xs_test) == N_test
    return xs_train, xs_test, ys_train, ys_test
        
def split(training_proportion, seed, xs, ys):
    training_indicator = np.random.uniform(size=len(xs)) < training_proportion
    return xs[training_indicator],ys[training_indicator],  xs[~training_indicator], ys[~training_indicator]

def get_data_helper(training_proportion, training_sampler, get_data_f, num_data, seed):

    # get_data
#    np.random.seed(seed)
    xs, ys = get_data_f(num_data)
    original_xs_shape, original_ys_shape = xs.shape, ys.shape
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit_transform(xs)
    xs_train, ys_train, xs_test, ys_test = split(training_proportion, seed, xs, ys)
    original_xs_train_shape, original_ys_train_shape = xs_train.shape, ys_train.shape
    original_xs_test_shape, original_ys_test_shape = xs_test.shape, ys_test.shape
    xs_train, ys_train = training_sampler(xs_train, ys_train)
    sampled_xs_train_shape, sampled_ys_train_shape = xs_train.shape, ys_train.shape

    return xs_train, xs_test, ys_train, ys_test

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
            
