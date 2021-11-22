# from imp import reload
import numpy as np
import mallows_kendall as mk
from scipy.spatial import distance
from scipy.stats import rankdata
import pandas as pd

def reverse(x): return x[::-1]

def is_duplicated(perm, sample):
  for p in sample:
    if np.array_equal(perm, p):
      return True
  return False

# Requires scipy 1.2.0
# from scipy.special import softmax
from scipy.special import logsumexp
def softmax(x, axis = None):
    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def binary_search_rho(w, ratio_samples_learn, weight_mass_learn,
                      # 0 <= w_i <= 1, w is sorted increasingly,
                      rho_ini=1, rho_end=0, tol=0.001):
  w = np.asarray(w)
  assert np.all(w >= 0.0)
  assert np.all(w <= 1.0)

  # If pos is None we take the largest 4th.
  # Find the rho s.t. the largest 25%(ratio_samples) of the weights  (rho**ws) take the 0.9(weight_mass) of the total ws.  rho^w[:pos] = 0.9*rho^w
  # codes as a recursive binary search in (0,1)
  pos = int(len(w) * ratio_samples_learn)
  rho_med = (rho_ini + rho_end) / 2
  # If the interval is very narrow, just return the value.
  if abs(rho_ini - rho_end) < 1e-20:
    return rho_med

  try:
      acum = np.cumsum(rho_med ** w)
      a = acum[pos]
      b = acum[-1]
      # If b is very small, all values are equal, the value of rho does not matter. Let's return 1.0
      if b < tol:
        return 1.0
      # If the differenc eot the target weight_mass is very small, just return.
      if abs(a / b - weight_mass_learn) < tol:
          return rho_med

      if a / b > weight_mass_learn:
          mid, last = rho_ini, rho_med
      else:
          mid, last = rho_med, rho_end
      return binary_search_rho(w, ratio_samples_learn, weight_mass_learn, mid, last)
  except: # MANUEL: How can the above fail?
       print(w)
       pos = int(len(w) * ratio_samples_learn)
       print(pos,len(w),ratio_samples_learn)
       rho_med = rho_ini + (rho_end - rho_ini) / 2
       acum = np.cumsum(rho_med ** w)
       a = acum[pos]
       b = acum[-1]
       print(f"binary_search_rho: a={a} b={b} a/b={a/b} wml={weight_mass_learn} rho_med={rho_med} rho_ini={rho_ini} rho_end={rho_end} w={w}")
       raise

def get_expected_distance(iterat, n, budget):
  # MANUEL: Should this be Kendall max dist?
  N = (n - 1) * n / 2
  f_ini, f_end = N / 4, 1
  iter_decrease = budget - 10 # MANUEL: Why 10?
  jump = (f_ini - f_end) / iter_decrease
  a = f_ini - jump * iterat
  return max(a, f_end)

def remove_duplicates(s):
  d = {a.tostring(): a for a in s}
  return list(d.values())

def design_random(m, n):
  """
  m: number of permutations to generate
  n: permutation size"""
  return remove_duplicates([ np.random.permutation(n) for _ in range(m)])

def min_distance(x, s, dist_fun):
  return np.apply_along_axis(dist_fun, -1, np.asarray(s), b=x).min()
  
def design_maxmindist(m, n, distance, budget = 1000, x0 = None):
  if x0 is None:
    sample = [ np.random.permutation(n) ]
  else:
    if not isinstance(x0, list):
      x0 = [ x0 ]
    sample += x0
    m -= len(sample) - 1
    
  while len(sample) < m:
    best = np.random.permutation(n)
    best_d = min_distance(best, sample, distance)
    for i in range(budget):
      xnew = np.random.permutation(n)
      xnew_d = min_distance(xnew, sample, distance)
      if xnew_d > best_d:
        best, best_d = xnew, xnew_d
    sample.append(best)
  return remove_duplicates(sample)


  
def UMM(instance, seed, budget, m_ini, eval_ranks, init,
        ratio_samples_learn = 0.1, weight_mass_learn = 0.9):

    np.random.seed(seed)
    if eval_ranks: # If True, the objective function works with ranks
      # FIXME: Do we really need this lambda?
      f_eval = lambda p: instance.fitness(p)
    else: # Otherwise, it works with orders
      f_eval = lambda p: instance.fitness(np.argsort(p))

    n = instance.n
    if init == "random":
      sample = design_random(m_ini, n)
      fitnesses = [f_eval(perm) for perm in sample]

    elif init == "maxmindist":
      sample = design_maxmindist(m_ini, n, distance = mk.distance)
      fitnesses = [f_eval(perm) for perm in sample]

    elif init == "greedy_euclidean":
      x0, f = instance.nearest_neighbor(np.full(n, -1, dtype=int), distance="euclidean")
      if not eval_ranks:
        x0 = np.argsort(x0)
      sample = design_maxmindist(m_ini, n, distance = mk.distance, x0 = x0)
      # avoid double evaluation
      fitnesses = [ f ] + [ f_eval(perm) for perm in sample[1:] ]

    else:
      raise ValueError(f"Invalid init: {init}")

    best_f = np.min(fitnesses)
    
    # ['rho','phi_estim','phi_sample','Distance']
    res = [ [np.nan, np.nan, np.nan,
             instance.distance_to_best(perm, mk.distance)] for perm in sample]

    #neighborhood = 1
    for m in range(budget - m_ini):
        ws = np.asarray(fitnesses).copy()
        # if neighborhood == 1: # Fast process the common case
        #   best_idx = np.argmin(ws)
        #   ws[:] = 0.
        #   ws[best_idx] = 1.
        # else:
        #   ws = 1. / rankdata(ws, method="min")
        #   ws[(-ws).argsort()[neighborhood:]] = 0.0
        #   print(f'fitnesses: {fitnesses}')
        #   print(f'ws       : {ws}')
        #   ws /= ws.sum()
        # rho = np.nan
        # FIXME: For maximization, this need to be changed.
        ws = ws - ws.min()
        # FIXME: Handle if ws.max() == 0.
        ws = ws / ws.max()
        co = ws.copy()
        co.sort()
        rho = binary_search_rho(co, ratio_samples_learn, weight_mass_learn)
        # rho = 1. / len(ws)
        # ws = rankdata(ws, method="min") 
        # print(fitnesses)
        # print(ws)
        ws = rho ** ws #MINIMIZE
        # print(ws)
        # ws = rho ** (1-ws) #MAXIMIZE
        # print(ws,co[:int(len(co)/4)].sum(),co.sum())
        # rho = 0
        # beta = 1 / 0.001
        #beta = len(ws) / m_ini # smlen
        # Round to avoid numerical instabilities with numbers close to zero.
        #ws = np.round(softmax(-beta * ws), 10) # MINIMIZE
        #inv_sample = sample
        # worst = np.argmax(ws)
        # worst_rev = reverse(sample[worst])
        # inv_sample = sample + [ worst_rev ]  # + [ reverse(p) for p in sample ]
        # ws = np.append(ws, 1)
        
        #ws = softmax(-np.asarray(fitnesses)) # MINIMIZE
        #ws = softmax(1. / 0.01 * np.hstack((-ws, ws-1))) # MINIMIZE
        #inv_sample = sample + [ reverse(p) for p in sample ]
        sigma0 = mk.weighted_median(np.asarray(sample), ws)
        #sigma0 = sample[np.argmin(fitnesses)]
        #phi_estim = mk.u_phi(inv_sample, sigma0, ws)
        # FIXME: We do not use phi_estim but it takes a significant amount of time to calculate it.
        phi_estim = np.nan
        expected_dist = get_expected_distance(m, n, budget)
        #expected_dist = 0
        phi_sample = mk.find_phi(n, expected_dist, expected_dist + 1)
        while True:
            perm = mk.sample(1, n, phi=phi_sample, s0 = sigma0)
            # dists = distance.cdist(perms, sample, metric=mk.kendallTau)
            # MANUEL: We probably do not need to sort, just find the min per axis=1.
            # dists = np.sort(dists, axis=1)
            # indi = np.argmax(dists[:, 0]) #index of the perm with the farthest closest permutation. Maximizes the min dist to the sample
            # FIXME: This should already be an array of int type.
            perm = np.asarray(perm, dtype='int')
            # Sample again if the permutation has already been evaluated.
            if not is_duplicated(perm, sample):
                break
                                
        for p in sample:
            assert not np.array_equal(perm, p), f"{perm} found in sample:\n {sample}"
        sample.append(perm)
        perm_f = f_eval(perm)
        fitnesses.append(perm_f)
        # if perm_f < best_f:
        #   best_f = perm_f
        #   neighborhood = 1
        # else:
        #   neighborhood += 1
              
        # print(f"UMM: eval={m}\tF={fitnesses[-1]}\tbest_known={instance.best_fitness}")
        # print(fitnesses,ws)

        # This is only used for reporting stats.
        res.append([rho, phi_estim, phi_sample, instance.distance_to_best(sigma0, mk.distance)])
    df = pd.DataFrame(res, columns=['rho','phi_estim','phi_sample','Distance'])
    df['Fitness'] = fitnesses
    df['x'] = [ ' '.join(map(str,s)) for s in sample ]
    df['m_ini'] = m_ini
    df['seed'] = seed
    df['budget'] = budget
    df['ratio_samples_learn'] = ratio_samples_learn
    df['weight_mass_learn'] = weight_mass_learn
    df['eval_ranks'] = eval_ranks
    df['init'] = init
    return df
