import numpy as np
import itertools as it
import scipy as sp
from mallows_model import *

def weighted_median(sample, ws):
    """
        Parameters
        ----------
        sample: numpy array
            RANKINGS
        ws: float
            weight of each permutation
        Returns
        -------
        ranking
            weigthed median ranking
    """
    return borda(sample * ws[:, None])


def max_dist(n):
    """
        Parameters
        ----------
        n: int
            length of permutations
        Returns
        -------
        int
            Maximum distance between permutations of given n length
    """
    return n * (n - 1) // 2 # Integer division

def compose(s, p):
    """This function composes two given permutations
    Parameters
    ----------
    s: ndarray
        The first permutation array
    p: ndarray
        The second permutation array
    Returns
    -------
    ndarray
        The composition of the permutations
    """
    return np.array(s[p])

def compose_partial(partial, full):
    """This function composes a partial permutation with an other (full)
        Parameters
        ----------
        partial: ndarray
            The partial permutation (should be filled with float)
        full:
            The full permutation (should be filled with integers)
        Returns
        -------
        ndarray
            The composition of the permutations
    """
    # MANUEL: If full contains np.nan, then it cannot be filled with integers, because np.nan is float.
    return [partial[i] if not np.isnan(i) else np.nan for i in full]

def inverse_partial(sigma):
    """This function computes the inverse of a given partial permutation
        Parameters
        ----------
        sigma: ndarray
            A partial permutation array (filled with float)
        Returns
        -------
        ndarray
            The inverse of given partial permutation
    """
    inv = np.full(len(sigma), np.nan)
    for i,j in enumerate(sigma):
        if not np.isnan(j):
            inv[int(j)] = i
    return inv

def inverse(s):
    """This function computes the inverse of a given permutation
        Parameters
        ----------
        s: ndarray
            A permutation array
        Returns
        -------
        ndarray
            The inverse of given permutation
    """
    return np.argsort(s)


def borda(rankings):
    """This function computes an average permutation given several permutations
        Parameters
        ----------
        rankings: ndarray
            Matrix of several permutations
        Returns
        -------
        ndarray
            The 'average' permutation of permutations given
    """
    # MANUEL: Using inverse instead of np.argsort clarifies the intention
    consensus = inverse( # give the inverse of result --> sigma_0
                            inverse( # give the indexes to sort the sum vector --> sigma_0^-1
                                        rankings.sum(axis=0) # sum the indexes of all permutations
                                        )
                            ) #borda
    return consensus

def borda_partial(rankings, w, k):
    """
        Parameters
        ----------
        Returns
        -------
    """
    a, b = rankings, w
    a, b = np.nan_to_num(rankings,nan=k), w
    aux = a * b
    borda = np.argsort(np.argsort(np.nanmean(aux, axis=0))).astype(float)
    mask = np.isnan(rankings).all(axis=0)
    borda[mask]=np.nan
    return borda




def expected_dist_mm(n, theta=None, phi=None):
    """Compute the expected distance, MM under the Kendall's-tau distance
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)
        Returns
        -------
        float
            The expected disance under the MMs
    """
    theta, phi = check_theta_phi(theta, phi)
    # MANUEL:
    # rnge = np.array(range(1,n+1))
    rnge = np.arange(1, n + 1)
    # exp_j_theta = np.exp(-j * theta)
    # exp_dist = (n * n.exp(-theta) / (1 - n.exp(-theta))) - np.sum(j * exp_j_theta / (1 - exp_j_theta)
    expected_dist = n * np.exp(-theta) / (1-np.exp(-theta)) - np.sum(rnge * np.exp(-rnge*theta) / (1 - np.exp(-rnge*theta)))

    return expected_dist

def variance_dist_mm(n, theta=None, phi=None):
    """
        Parameters
        ----------
        Returns
        -------
    """
    theta, phi = check_theta_phi(theta, phi)
    rnge = np.array(range(1,n+1))
    variance = (phi*n)/(1-phi)**2 - np.sum((pow(phi,rnge) * rnge**2)/(1-pow(phi,rnge))**2)

    return variance

def expected_v(n, theta=None, phi=None, k=None):#txapu integrar
    """This function computes the expected decomposition vector
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)
        k: int
            Index to which the decomposition vector is needed ???
        Returns
        -------
        ndarray
            The expected decomposition vector
    """
    theta, phi = check_theta_phi(theta, phi)
    if k is None: k = n-1
    if type(theta)!=list: theta = np.full(k, theta)
    rnge = np.array(range(k))
    expected_v = np.exp(-theta[rnge]) / (1-np.exp(-theta[rnge])) - (n-rnge) * np.exp(-(n-rnge)*theta[rnge]) / (1 - np.exp(-(n-rnge)*theta[rnge]))
    return expected_v

def variance_v(n, theta=None, phi=None, k=None):
    """
        Parameters
        ----------
        Returns
        -------
    """
    theta, phi = check_theta_phi(theta, phi)
    if k is None:
        k = n-1
    if type(phi)!=list:
        phi = np.full(k, phi)
    rnge = np.array(range(k))
    var_v = phi[rnge]/(1-phi[rnge])**2 - (n-rnge)**2 * phi[rnge]**(n-rnge) / (1-phi[rnge]**(n-rnge))**2
    return var_v

def expected_dist_top_k(n, k, theta=None, phi=None):
    """Compute the expected distance for top-k rankings, following
    a MM under the Kendall's-tau distance
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)
        Returns
        -------
        float
            The expected disance under the MMs
    """
    theta, phi = check_theta_phi(theta, phi)
    rnge = np.array(range(n-k+1,n+1))
    expected_dist = k * np.exp(-theta) / (1-np.exp(-theta)) - np.sum(rnge * np.exp(-rnge*theta) / (1 - np.exp(-rnge*theta)))
    return expected_dist

def variance_dist_top_k(n, k, theta=None, phi=None):
    """
        Compute the variance of the distance for top-k rankings, following
        a MM under the Kendall's-tau distance
        Parameters
        ----------
        Returns
        -------
    """
    theta, phi = check_theta_phi(theta, phi)
    rnge = np.array(range(n-k+1,n+1))
    variance = (phi*k)/(1-phi)**2 - np.sum((pow(phi,rnge) * rnge**2)/(1-pow(phi,rnge))**2)
    return variance


def psi_mm(n, theta=None, phi=None):
    """This function computes the normalization constant psi
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)
        Returns
        -------
        float
            The normalization constant psi
    """
    rnge = np.array(range(2,n+1))
    if theta is not None:
        return np.prod((1-np.exp(-theta*rnge))/(1-np.exp(-theta)))
    if phi is not None:
        return np.prod((1-np.power(phi,rnge))/(1-phi))
    theta, phi = check_theta_phi(theta, phi)

# def prob_mode(n, theta):
#     """This function computes the probability mode
#         Parameters for both Mallows and Generalized Mallows
#         ----------
#         n: int
#             Length of the permutation in the considered model
#         theta: float/int/list/numpy array (see theta, params)
#             Real dispersion parameter
#         Returns
#         -------
#         float
#             The probability mode
#     """
#     psi = (1 - np.exp(( - n + np.arange(n-1) )*(theta)))/(1 - np.exp( -theta))
#     psi = np.prod(psi)
#     return np.prod(1.0/psi)

def prob(sigma, sigma0, theta=None,phi=None):
    """This function computes the probability of a permutation given a distance to the consensus
        Parameters
        ----------
        n: int
            Length of the permutation in the considered model
        theta: float
            Dispersion vector
        dist: int
            Distance of the permutation to the consensus permutation
        Returns
        -------
        float
            Probability of the permutation
    """
    theta, phi = check_theta_phi(theta, phi)
    n = len(sigma)
    # rnge = np.array(range(n-1))
    psi = (1 - np.exp(( - n + np.arange(n-1) )*(theta)))/(1 - np.exp( -theta))
    psi = np.prod(psi)
    return np.exp(-theta * distance(sigma,sigma0)) / psi

def prob_sample(perms, sigma, theta=None, phi=None):
    """This function computes the probabilities for each permutation of a sample
    of several permutations
        Parameters
        ----------
        perms: ndarray
            The matrix of permutations
        sigma: ndarray
            Permutation mode
        theta: float
            Real dispersion parameter (optionnal if phi is given)
        phi: float
            Real dispersion parameter (optionnal if theta is given)
        Returns
        -------
        ndarray
            Array of probabilities for each permutation given as input
    """
    m, n = perms.shape
    theta, phi = check_theta_phi(theta, phi)
    rnge = np.array(range(n-1))
    psi = (1 - np.exp(( - n + rnge )*(theta)))/(1 - np.exp( -theta))
    psi = np.prod(psi)
    return np.array([np.exp(-theta*distance(perm, sigma)) / psi for perm in perms])

def fit_mm(rankings, s0=None):
    """This function computes the consensus permutation and the MLE for the
    dispersion parameter phi for MM models
        Parameters
        ----------
        rankings: ndarray
            The matrix of permutations
        s0: ndarray, optional
            The consensus permutation (default value is None)
        Returns
        -------
        tuple
            The ndarray corresponding to s0 the consensus permutation and the
            MLE for the dispersion parameter phi
    """
    m, n = rankings.shape
    if s0 is None: s0 = np.argsort(np.argsort(rankings.sum(axis=0))) #borda
    dist_avg = np.mean(np.array([distance(s0, perm) for perm in rankings]))
    try:
        theta = sp.optimize.newton(mle_theta_mm_f, 0.01, fprime=mle_theta_mm_fdev, args=(n, dist_avg), tol=1.48e-08, maxiter=500, fprime2=None)
    except:
        if dist_avg == 0.0:
            return s0, np.exp(-5)#=phi
        print("Error in function: fit_mm. dist_avg=",dist_avg, dist_avg == 0.0)
        print(rankings)
        print(s0)
        raise
    return s0, np.exp(-theta)#=phi
# def fit_mm_phi(n, dist_avg):
#     """Same as fit_mm but just returns phi ??? Also does not compute dist_avg
#     but take it as a parameter
#
#         Parameters
#         ----------
#         n: int
#             Dimension of the permutations
#         dist_avg: float
#             Average distance of the sample (between the consensus and the
#             permutations of the consensus)
#
#         Returns
#         -------
#         float
#             The MLE for the dispersion parameter phi
#     """
#     try:
#         theta = sp.optimize.newton(mle_theta_mm_f, 0.01, fprime=mle_theta_mm_fdev, args=(n, dist_avg), tol=1.48e-08, maxiter=500, fprime2=None)
#     except:
#         if dist_avg == 0.0:
#             return s0, np.exp(-5)#=phi
#         print("error. fit_mm. dist_avg=",dist_avg, dist_avg == 0.0)
#         print(rankings)
#         print(s0)
#         raise
#     # theta = - np.log(phi)
#     return np.exp(-theta)

def fit_gmm(rankings, s0=None):
    """This function computes the consensus permutation and the MLE for the
    dispersion parameters theta_j for GMM models
        Parameters
        ----------
        rankings: ndarray
            The matrix of permutations
        s0: ndarray, optional
            The consensus permutation (default value is None)
        Returns
        -------
        tuple
            The ndarray corresponding to s0 the consensus permutation and the
            MLE for the dispersion parameters theta
    """
    m, n = rankings.shape
    if s0 is None:
        s0 = np.argsort(np.argsort(rankings.sum(axis=0))) #borda
    V_avg = np.mean(np.array([ranking_to_v(sigma)[:-1] for sigma in rankings]), axis = 0)
    try:
        theta = []
        for j in range(1, n):
            theta_j = sp.optimize.newton(mle_theta_j_gmm_f, 0.01, fprime=mle_theta_j_gmm_fdev, args=(n, j, V_avg[j-1]), tol=1.48e-08, maxiter=500, fprime2=None)
            theta.append(theta_j)
    except:
        print("Error in function fit_gmm")
        raise
    return s0, theta




def mle_theta_mm_f(theta, n, dist_avg):
    """Computes the derivative of the likelihood
    parameter
        Parameters
        ----------
        theta: float
            The dispersion parameter
        n: int
            Dimension of the permutations
        dist_avg: float
            Average distance of the sample (between the consensus and the
            permutations of the consensus)
        Returns
        -------
        float
            Value of the function for given parameters
    """
    aux = 0
    rnge = np.array(range(1,n))
    aux = np.sum((n-rnge+1)*np.exp(-theta*(n-rnge+1))/(1-np.exp(-theta*(n-rnge+1))))
    aux2 = (n-1) / (np.exp( theta ) - 1) - dist_avg

    return aux2 - aux

def mle_theta_mm_fdev(theta, n, dist_avg):
    """This function computes the derivative of the function mle_theta_mm_f
    given the dispersion parameter and the average distance
        Parameters
        ----------
        theta: float
            The dispersion parameter
        n: int
            The dimension of the permutations
        dist_avg: float
            Average distance of the sample (between the consensus and the
            permutations of the consensus)
        Returns
        -------
        float
            The value of the derivative of function mle_theta_mm_f for given
            parameters
    """
    aux = 0
    rnge = np.array(range(1, n))
    aux = np.sum((n-rnge+1)*(n-rnge+1)*np.exp(-theta*(n-rnge+1))/pow((1 - np.exp(-theta * (n-rnge+1))), 2))
    aux2 = (- n + 1) * np.exp( theta ) / pow ((np.exp( theta ) - 1), 2)

    return aux2 + aux

def mle_theta_j_gmm_f(theta_j, n, j, v_j_avg):
    """Computes the derivative of the likelihood
    parameter theta_j in the GMM
        Parameters
        ----------
        theta: float
            The jth dispersion parameter theta_j
        n: int
            Dimension of the permutations
        j: int
            The position of the theta_j in vector theta of dispersion parameters
        v_j_avg: float
            jth element of the average decomposition vector over the sample
        Returns
        -------
        float
            Value of the function for given parameters
    """
    f_1 = np.exp( -theta_j ) / ( 1 - np.exp( -theta_j ) )
    f_2 = - ( n - j + 1 ) * np.exp( - theta_j * ( n - j + 1 ) ) / ( 1 - np.exp( - theta_j * ( n - j + 1 ) ) )
    return f_1 + f_2 - v_j_avg

def mle_theta_j_gmm_fdev(theta_j, n, j, v_j_avg):
    """This function computes the derivative of the function mle_theta_j_gmm_f
    given the jth element of the dispersion parameter and the jth element of the
    average decomposition vector
        Parameters
        ----------
        theta: float
            The jth dispersion parameter theta_j
        n: int
            Dimension of the permutations
        j: int
            The position of the theta_j in vector theta of dispersion parameters
        v_j_avg: float
            jth element of the average decomposition vector over the sample
        Returns
        -------
        float
            The value of the derivative of function mle_theta_j_gmm_f for given
            parameters
    """
    fdev_1 = - np.exp( - theta_j ) / pow( ( 1 - np.exp( -theta_j ) ), 2 )
    fdev_2 = pow( n - j + 1, 2 ) * np.exp( - theta_j * ( n - j + 1 ) ) / pow( 1 - np.exp( - theta_j * ( n - j + 1 ) ), 2 )
    return fdev_1 + fdev_2

def likelihood_mm(perms, s0, theta):
    """This function computes the log-likelihood for MM model given a matrix of
    permutation, the consensus permutation, and the dispersion parameter
        Parameters
        ----------
        perms: ndarray
            A matrix of permutations
        s0: ndarray
            The consensus permutation
        theta: float
            The dispersion parameter
        Returns
        -------
        float
            Value of log-likelihood for given parameters
    """
    m,n = perms.shape
    rnge = np.array(range(2,n+1))
    psi = 1.0 / np.prod((1-np.exp(-theta*rnge))/(1-np.exp(-theta)))
    probs = np.array([np.log(np.exp(-distance(s0, perm)*theta)/psi) for perm in perms])
    # print(probs,m,n)
    return probs.sum()

def sample(m, n, k=None, theta=None, phi=None, s0=None):
    """This function generates m permutations (rankings) according
    to Mallows Models (if the given parameters are m, n, k/None,
    theta/phi: float, s0/None) or Generalized Mallows Models
    (if the given parameters are m, theta/phi: ndarray, s0/None).
    Moreover, the parameter k allows the function to generate top-k rankings only.
        Parameters
        ----------
        m: int
            The number of rankings to generate
        theta: float or ndarray, optional (if phi given)
            The dispersion parameter theta
        phi: float or ndarray, optional (if theta given)
            The dispersion parameter phi
        k: int
            number of known positions of items for the rankings
        s0: ndarray
            The consensus ranking
        Returns
        -------
        list
            The rankings generated
    """
    if k is not None and n is None:
        # MANUEL: If we don't raise an error the program continues which makes debugging difficult.
        raise ValueError("Error, n is not given!")

    theta, phi = check_theta_phi(theta, phi)

    if n is not None: #TODO, n should be always given
        theta = np.full(n-1, theta)

    n = len(theta) + 1 #TODO, n should be always given

    if s0 is None:
        s0 = np.array(range(n))

    rnge = np.arange(n - 1)

    psi = (1 - np.exp(( - n + rnge )*(theta[ rnge ])))/(1 - np.exp( -theta[rnge]))
    vprobs = np.zeros((n,n))
    for j in range(n-1):
        vprobs[j][0] = 1.0/psi[j]
        for r in range(1,n-j):
            vprobs[j][r] = np.exp( -theta[j] * r ) / psi[j]
    sample = []
    vs = []
    for samp in range(m):
        v = [np.random.choice(n,p=vprobs[i,:]) for i in range(n-1)]
        v += [0]
        ranking = v_to_ranking(v, n)
        sample.append(ranking)

    sample = np.array([s[s0] for s in sample])

    if k is not None:
        sample_rankings = np.array([inverse(ordering) for ordering in sample])
        sample_rankings = np.array([ran[s0] for ran in sample_rankings])
        sample = np.array([[i if i in range(k) else np.nan for i in ranking] for
                        ranking in sample_rankings])
    return sample.squeeze()

def v_to_ranking(v, n):
    """This function computes the corresponding permutation given
    a decomposition vector
        Parameters
        ----------
        v: ndarray
            Decomposition vector, same length as the permutation, last item must be 0
        n: int
            Length of the permutation
        Returns
        -------
        ndarray
            The permutation corresponding to the decomposition vectors
    """
    rem = list(range(n))
    rank = np.full(n, np.nan)
    for i in range(len(v)):
        rank[i] = rem[v[i]]
        rem.pop(v[i])
    return rank

def ranking_to_v(sigma, k=None):
    """This function computes the corresponding decomposition vector given
    a permutation
        Parameters
        ----------
        sigma: ndarray
            A permutation
        k: int, optionnal
            The index to perform the conversion for a partial
            top-k list
        Returns
        -------
        ndarray
            The decomposition vector corresponding to the permutation. Will be
            of length n and finish with 0
    """
    n = len(sigma)
    if k is not None:
        sigma = sigma[:k]
        sigma = np.concatenate((sigma,np.array([np.float(i) for i in range(n) if i not in sigma])))
    V = []
    for j, sigma_j in enumerate(sigma):
        V_j = 0
        for i in range(j+1,n):
            if sigma_j > sigma[i]:
                V_j += 1
        V.append(V_j)
    return np.array(V)
# def discordances_to_permut(indCode, refer):
#     """
#         Parameters
#         ----------
#         Returns
#         -------
#     """
#     print("warning. discordances_to_permut is deprecated. Use function v_to_ranking")
#     return v_to_ranking(indCode)


def count_inversion(left, right):
    """
    This function use merge sort algorithm to count the number of
    inversions in a permutation of two parts (left, right).
    Parameters
    ----------
    left: ndarray
        The first part of the permutation
    right: ndarray
        The second part of the permutation
    Returns
    -------
    result: ndarray
        The sorted permutation of the two parts
    count: int
        The number of inversions in these two parts
    """
    result = []
    count = 0
    i, j = 0, 0
    left_len = len(left)
    while i < left_len and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            count += left_len - i
            j += 1
    result += left[i:]
    result += right[j:]

    return result, count

def mergeSort_rec(lst):
    """
    This function count the number of inversions in a permutation by calling
    count_inversion recursively.
    Parameters
    ----------
    lst: ndarray
        The permutation
    Returns
    -------
    result: ndarray
        The sorted permutation
    (a + b + c): int
        The number of inversions
    """
    lst = list(lst)
    if len(lst) <= 1:
        return lst, 0
    middle = int( len(lst) / 2 )
    left, a   = mergeSort_rec(lst[:middle])
    right, b  = mergeSort_rec(lst[middle:])
    result, c = count_inversion(left, right)
    return result, (a + b + c)


# MANUEL: Should this be an integer?
def dist_at_uniform(n): return (n - 1) * n / 4

# MANUEL: I'm not going to change it but I think this interface is error-prone
# because if b is None because of some bug, it seems to work correctly. A
# better interface would have another function, distance_to_identity(a).
def distance(a, b=None):
    """
    This function computes the kendall's-tau distance between two permutations
    using merge sort algorithm.
    If only one permutation is given, the distance will be computed with the
    identity permutation as the second permutation
   Parameters
   ----------
   A: ndarray
        The first permutation
   B: ndarray, optionnal
        The second permutation (default is None)
   Returns
   -------
   int
        The kendall's-tau distance between both permutations
    """
    n = len(a)
    if b is None:
        b = np.arange(n)
    a = np.asarray(a)
    b = np.asarray(b)

    # check if A contains NaNs
    msk = np.isnan(a)
    if msk.any(): # A contains NaNs
        indexes = np.arange(n)[msk]
        # reverse the indexes
        indexes = sorted(indexes, reverse=True)
        for i in indexes: # delete all NaNs and their associated values in B
            a = np.delete(a, i)
            b = np.delete(b, i)

    # check if B contains NaNs
    msk = np.isnan(b)
    if msk.any(): # B contains NaNs
        indexes = np.arange(n - len(indexes))[msk]
        # reverse the indexes
        indexes = sorted(indexes, reverse=True)
        for i in indexes: # delete all NaNs and their associated values in A
            # MANUEL: You can pass all indexes at once and avoid the for-loop
            a = np.delete(a, i)
            b = np.delete(b, i)

    inverse = np.argsort(b)
    compose = a[inverse]
    _, distance = mergeSort_rec(compose)
    return distance


# def dist_alpha(alpha, k):
#     """Compute the distance of a partial ordering (also called top-k list)
#     according to an alternative definition of Kendall's-tau distance. The
#     distance is defined as follows: it is the sum of the js in the head larger
#     than i for every i.
#
#         Parameters
#         ----------
#         alpha: ndarray
#             The partial ordering
#         k: int
#             The order ??? of the partial list
#
#         Returns
#         -------
#         int
#             The kendall's-tau distance alternative for alpha
#     """
#     # an alternative def for kendall is to sum the js in the tail smaller than i, for every i
#     # or the js in the head larger than i for every i*.
#     # we take this since the head is defined an d the tail is not for alpha in Alpha
#     dist = 0
#     for j in range(k):
#         dist += alpha[j] - np.sum([1 for i in alpha[:j] if i<alpha[j]])
#
#     return dist
# def dist_beta(beta, sigma=None):
#     """Compute the distance of a partial ranking according to an alternative
#     definition of Kendall's-tau distance. The distance is defined as follows:
#     missing ranks in beta are filled with a value greater than all the values
#     in both rankings (length of the rankings + 1 here). Then the classical
#     Kendall's-tau distance is applied to this new vector.
#
#         Parameters
#         ----------
#         beta: ndarray
#             The partial ranking
#         sigma: ndarray, optional
#             A full permutation to which wew want to compute the distance with
#             beta (default None, sigma will be the identity permutation)
#
#         Returns
#         -------
#         int
#             The kendall's-tau distance alternative for beta
#     """
#     n = len(beta)
#     if sigma is None:
#         sigma = list(range(n))
#     aux = beta.copy()
#     aux = [i if not np.isnan(i) else n+1 for i in aux ]
#     return distance(aux, sigma)

def p_distance(beta_1, beta_2, k, p=0):
    alpha_1 = beta_to_alpha(beta_1, k=k)
    alpha_2 = beta_to_alpha(beta_2, k=k)
    d = 0
    p_counter = 0
    alpha_1Ualpha_2 = list(set(int(x) for x in np.union1d(alpha_1, alpha_2) if np.isnan(x) == False))
    for i_index, i in enumerate(alpha_1Ualpha_2):
        i_1_nan = np.isnan(beta_1[i])
        i_2_nan = np.isnan(beta_2[i])
        for j in alpha_1Ualpha_2[i_index + 1:] :
            j_1_nan = np.isnan(beta_1[j])
            j_2_nan = np.isnan(beta_2[j])
            if not i_1_nan and  not j_1_nan and not i_2_nan and not j_2_nan:
                if ( beta_1[i] > beta_1[j] and beta_2[i] > beta_2[j] ) or \
                ( beta_1[i] < beta_1[j] and beta_2[i] < beta_2[j] ):
                    continue
                elif ( beta_1[i] > beta_1[j] and beta_2[i] < beta_2[j] ) or \
                ( beta_1[i] < beta_1[j] and beta_2[i] > beta_2[j] ):
                    d += 1
            elif ( not i_1_nan and not j_1_nan and ( (not i_2_nan and j_2_nan) or (i_2_nan and not j_2_nan) ) ) or \
            ( not i_2_nan and not j_2_nan and ( (not i_1_nan and j_1_nan) or (i_1_nan and not j_1_nan) ) ):
                if i_1_nan:
                    d += int(beta_2[j] > beta_2[i])
                elif j_1_nan:
                    d += int(beta_2[i] > beta_2[j])
                elif i_2_nan:
                    d += int(beta_1[j] > beta_1[i])
                elif j_2_nan:
                    d += int(beta_1[i] > beta_1[j])
            elif ( not i_1_nan and j_1_nan and i_2_nan and not j_2_nan ) or \
            ( i_1_nan and not j_1_nan and not i_2_nan and j_2_nan ):
                d += 1
            elif ( not i_1_nan and not j_1_nan and i_2_nan and j_2_nan ) or \
            ( i_1_nan and j_1_nan and not i_2_nan and not j_2_nan ):
                p_counter += 1
    return d + p_counter*p

def alpha_to_beta(alpha,k): #aux for the p_distance
    inv = np.full(len(alpha), np.nan)
    for i,j in enumerate(alpha[:k]):
        inv[int(j)] = i
    return inv
def beta_to_alpha(beta,k): #aux for the p_distance
    inv = np.full(len(beta), np.nan)
    for i,j in enumerate(beta):
        if not np.isnan(j):
            inv[int(j)] = i
    return inv


def num_perms_at_dist(n):
    """This function computes the number of permutations of length 1 to n for
    each possible Kendall's-tau distance d
        Parameters
        ----------
        n: int
            Dimension of the permutations
        Returns
        -------
        ndarray
            ??? ---> to finish
    """
    sk = np.zeros((n+1,int(n*(n-1)/2+1)))
    for i in range(n+1):
        sk[i,0] = 1
    for i in range(1,1+n):
        for j in range(1,int(i*(i-1)/2+1)):
            if j - i >= 0 :
                sk[i,j] = sk[i,j-1]+ sk[i-1,j] - sk[i-1,j-i]
            else:
                sk[i,j] = sk[i,j-1]+ sk[i-1,j]
    return sk.astype(np.uint64)

def random_perm_at_dist(n, dist, sk):
    """
        Parameters
        n, dist
        sk, the matrix restured by the function 'num_perms_at_dist(n)'
        ----------
        Returns
        -------
    """
    # param sk is the results of the function num_perms_at_dist(n)
    i = 0
    probs = np.zeros(n+1)
    v = np.zeros(n,dtype=int)
    while i<n and dist > 0 :
        rest_max_dist = (n - i - 1 ) * ( n - i - 2 ) / 2
        if rest_max_dist  >= dist:
            probs[0] = sk[n-i-1,dist]
        else:
            probs[0] = 0
        mi = min(dist + 1 , n - i )
        for j in range(1,mi):
            if rest_max_dist + j >= dist: probs[j] = sk[n-i-1, dist-j]
            else: probs[ j ] = 0
        v[i] = np.random.choice(mi,1,p=probs[:mi]/probs[:mi].sum())
        dist -= v[i]
        i += 1
    return v_to_ranking(v,n)

def find_phi_n(n, bins):
    ed, phi_ed = [], []
    ed_uniform = (n*(n-1)/2)/2
    for dmin in np.linspace(0,ed_uniform-1,bins):
        ed.append(dmin)
        phi_ed.append(find_phi(n, dmin, dmin+1))
    return ed, phi_ed

# MANUEL: You had a comment before explaining what this function does.
def find_phi(n, dmin, dmax):
    assert dmin < dmax
    imin, imax = np.float64(0), np.float64(1)
    iterat = 0
    while iterat < 500:
        med = (imax + imin) / 2
        # MANUEL: If expected_dist_mm accepts both phi and theta, why convert?
        d = expected_dist_mm(n, theta = phi_to_theta(med))
        if d < dmin: imin = med
        elif d > dmax: imax = med
        else: return med
        iterat += 1
    # MANUEL: This function can stop without returning anything, which will
    # lead to a bug. Let's make sure we give an error.
    assert False, "Max iterations reached"


# TODO, move to MM and merge with find phi
def find_proba_mode(n, target_prob, tol=1e-10, maxiter=1000):
    # imax, imin, med: vlalues for phi
    imin, imax = np.float64(0), np.float64(1)
    iterat = 0
    while iterat < 500:
        med = (imax + imin) / 2
        p = prob(np.arange(n), np.arange(n), theta=None,phi=med)
        if iterat%20==0: print("trace find proba", iterat, abs(p - target_prob))
        if abs(p - target_prob) < tol:return med
        if p > target_prob: imin = med
        else : imax = med

        iterat += 1
    # MANUEL: This function can stop without returning anything, which will
    # lead to a bug. Let's make sure we give an error.
    assert False, "Max iterations reached"


# end
