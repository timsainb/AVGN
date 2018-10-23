# mutual information
import scipy
import scipy.special
from sklearn.metrics.cluster.supervised import contingency_matrix
from scipy import sparse as sp
### resampling
from scipy import interpolate
# modelling
from lmfit import Model
from scipy import signal
import lmfit


def est_entropy(X):

    dist = np.array(X)
    N = float(len(dist))
    Nall = [np.sum(dist == c) for c in set(dist)]
    pAll = np.array([float(Ni) * scipy.special.psi(float(Ni)) for Ni in Nall])
    S_hat = np.log2(N) - 1./N * np.sum(pAll)
    var = np.var(scipy.special.psi(np.array(Nall, dtype='float32'))/N)
    return S_hat, var

def est_mutual_info(a,b):
    e_a, var_a = est_entropy(a)
    e_b, var_b = est_entropy(b)
    e_ab, var_ab = est_joint_entropy(a,b)
    return e_a + e_b - e_ab, var_a + var_b + var_ab

def est_joint_entropy(labels_true, labels_pred):

    N = float(len(labels_true))
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    nzx, nzy, Nall = sp.find(contingency)

    pAll = np.array([Ni * scipy.special.psi(Ni) for Ni in Nall if Ni > 0])
    S_hat = np.log2(N) - 1/N * np.sum(pAll)
    return S_hat, np.var(1./N * pAll)

def MI_from_distributions(sequences, dist):
    # create distributions
    if np.sum([len(seq) > dist for seq in sequences]) == 0:
        return (np.nan, np.nan)
    distribution_a=np.concatenate([seq[dist:] for seq in sequences if len(seq) > dist])
    distribution_b=np.concatenate([seq[:-dist] for seq in sequences if len(seq) > dist])
    # calculate MI
    return est_mutual_info(distribution_a, distribution_b)

def sequential_mutual_information(sequences, distances, n_jobs = 1, verbosity = 5):
    """
    Compute mutual information as a function of distance between sequences
    if n_jobs > 1, we will run in parallel
    """
    n_seqs = len(sequences)
    shuffled_sequences = [np.random.permutation(i) for i in sequences] # lower bound

    if n_jobs == 1:
        MI = [MI_from_distributions(sequences, dist) for dist_i, dist in enumerate(tqdm(distances, leave=False))]
        shuff_MI = [MI_from_distributions(shuffled_sequences, dist) for dist_i, dist in enumerate(tqdm(distances, leave=False))]
    else:
        MI = parallel(delayed(MI_from_distributions)(sequences, dist) for dist_i, dist in enumerate(tqdm(distances, leave=False)))
        shuff_MI = parallel(delayed(MI_from_distributions)(shuffled_sequences, dist) for dist_i, dist in enumerate(tqdm(distances, leave=False)))

    return np.array(MI).T, np.array(shuff_MI).T
