# any snippits of code that don't fit elsewhere

import numpy as np
import pickle 

def zero_one_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def save_dict_pickle(dict_, save_loc):
    with open(save_loc, 'wb') as f:
        pickle.dump(dict_, f, protocol=pickle.HIGHEST_PROTOCOL)

def rescale(X, out_min, out_max):
    return out_min + (X - np.min(X)) * ((out_max - out_min) / (np.max(X) - np.min(X)))