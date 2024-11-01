T = 10 # mins
lam = 2 # customers/min
n = 1000

h = T/n # subinterval length
p = lam * h # prob. of arrival in each subinterval

n_bins = n // 10 # number of bins in histograms

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import poisson
from tqdm import tqdm

def main():
    n_runs = 1e5

    all_variables = defaultdict(list)
    for i in tqdm(range(int(n_runs))):
        # n_arrivals = 0
        # while n_arrivals == 0: # repeat the experiment if there is no arrival
        arrivals = (np.random.rand(n,) < p)
        # print('arrivals: ', arrivals)
        n_arrivals = np.sum(arrivals)

        first_arrival_time = h * np.where(arrivals)[0][0] if len(np.where(arrivals)[0]) > 0 else np.nan
        second_arrival_time = h * np.where(arrivals)[0][1] if len(np.where(arrivals)[0]) > 1 else np.nan

        all_variables['n_arrivals'].append(n_arrivals)
        all_variables['first_arrival_time'].append(first_arrival_time)
        all_variables['second_arrival_time'].append(second_arrival_time)


    fig, axs = plt.subplots(1, 2, figsize = (12, 6))


    ### First arrival T_1 \sim Exp(lam) ###
    axs[0].hist(x = np.array(all_variables['first_arrival_time']), bins=n_bins, density = True, label = 'Sim.')

    t = np.linspace(0, T, n // 10)
    exp_t = lam * np.exp(-lam * t)
    axs[0].plot(t, exp_t, '--r', label = 'Exp. p.d.f.')
    # axs[0].set_xlabel('t')
    axs[0].set_xlabel(r'$T_1$')
    axs[0].grid(True)
    axs[0].legend(loc = 'best')


    ### Number of arrivals N(T) \sim Poisson(\lam T) ###
    k = np.arange(0, np.array(all_variables['n_arrivals']).max() + 1)
    axs[1].hist(x = np.array(all_variables['n_arrivals']), bins = np.arange(start=0, stop=k[-1] + 1), density = True, label = 'Sim.')

    pois_k = poisson.pmf(k=k, mu = lam * T)
    axs[1].plot(k, pois_k, '--r', label = 'Poisson p.m.f.')

    # axs[1].set_xlabel('k')
    axs[1].set_xlabel(r'$N(T)$')
    axs[1].grid(True)
    axs[1].legend(loc = 'best')


    ### Order statistics ###
    m = (T * lam)

    # print('m: ', m)
    # print('n_arrivals: ', all_variables['n_arrivals'])
    
    run_idx = np.array(all_variables['n_arrivals']) == int(m)
    print(f'Fraction of runs with N(T) = {m} arrivals: ', np.sum(run_idx) / len(run_idx))

    print('Order statistic 1: ', np.array(all_variables['first_arrival_time'])[run_idx].mean())
    print('Order statistic 2: ', np.array(all_variables['second_arrival_time'])[run_idx].mean())
    print('T / (m+1) = ', T / (m+1))


    plt.show()




















if __name__ == "__main__":
    main()