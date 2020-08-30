import numpy as np, matplotlib.pyplot as plt, pandas as pd
import pymc as pm, scipy

def my_convolve(a, v):
    """Keep the length of `a`, but start with \sum_s a[0-s]*v[s]
    """
    b = np.convolve(a, v, mode='full')[:len(a)]
    return b


def my_weibull_kernel(amp, a, c):
    rv = scipy.stats.exponweib(a=a, c=c)

    # prevent underflow errors 
    x_min = min(21, int(np.ceil(rv.ppf(0.01))))
    x_max = min(21, int(np.floor(rv.ppf(0.99))))
    
    kernel_vals = amp * rv.pdf(range(x_min, x_max))
    return np.concatenate([np.zeros(x_min), kernel_vals, np.zeros(21-x_max)])


def apply_day_of_week_effect(s, beta, dow):
    """find day of week for s and multiply s[i] by beta[dow[i]]
    s : array-like to change
    beta : array-like to use for change
    dow : array-like indicating what day-of-week each entry of s is
    beta[0] = monday effect
    beta[6] = sunday effect
    """
    assert len(beta) == 7, 'expect effects for 7 days in the week'
    assert len(s) == len(dow), 'expect a dow for every entry of s'
    s_beta = beta[dow]
    t = s * np.exp(s_beta)
    return t


def simulate(s_infections_true, n_pop, symp_kernel_true, cc_kernel_true, cc_day_of_week_effects_true):
    df = pd.DataFrame(index=s_infections_true.index)
    
    df['n_pop'] = n_pop
    df['n_sampled'] = 1_000
    
    symp_intensity = my_convolve(s_infections_true, symp_kernel_true)
    df['n_symp'] = np.random.poisson(1e-3 + symp_intensity/df.n_pop * df.n_sampled)
    
    cc_intensity_smooth = my_convolve(s_infections_true, cc_kernel_true)
    dow = s_infections_true.index.map(pd.Timestamp.weekday).values
    cc_intensity = apply_day_of_week_effect(cc_intensity_smooth, cc_day_of_week_effects_true, dow)
    df['n_cc'] = np.random.poisson(cc_intensity)
                                     
    return df.iloc[21:]  # skip first three week


def model(data):
    X_smooth = [pm.Uniform(f'X_smooth_{i}', 0, data.n_pop, value=10*data.n_cc.iloc[i]) for i in range(len(data))]
    
    # potential to smooth X with
    # some appropriate smoothing prior
    @pm.potential
    def X_smooth_potential(X_smooth=X_smooth):
        mu = np.diff(np.log(X_smooth), n=1)
        return pm.normal_like(np.zeros_like(mu), mu, 500)
      
    # day of week effects 
    dow = data.index.map(pd.Timestamp.weekday).values
    beta_dow_infection = np.zeros(7)
    beta_dow_cc_nonzero = pm.Normal('beta_dow_cc_nonzero', 0, 100, value=np.zeros(4))
    @pm.deterministic(trace=True)
    def beta_dow_cc(beta_dow_cc_nonzero=beta_dow_cc_nonzero):
        return np.array([beta_dow_cc_nonzero[0], 0, 0, 0,
                         beta_dow_cc_nonzero[1], beta_dow_cc_nonzero[2], beta_dow_cc_nonzero[3]])

    # latent state: number of new infections each day
    @pm.deterministic(trace=True)
    def X(X_smooth=X_smooth, beta_dow_infection=beta_dow_infection):
        return apply_day_of_week_effect(
            X_smooth, beta_dow_infection, dow)
        
    # symptom kernel parameterized by Weibull distribution
    symp_kernel_amp = pm.Uniform('symp_kernel_amp', .1, 10, value=5)
    symp_kernel_knots = [0] + [pm.Uniform(f'symp_kernel_knot_{i}', 0, .1, value=.01) for i in range(6)]
    @pm.deterministic(trace=True)
    def symp_kernel(symp_kernel_amp=symp_kernel_amp, symp_kernel_knots=symp_kernel_knots):
        ii = np.linspace(0,7,num=21, endpoint=False)
        ii = np.array(np.floor(ii), dtype=int)
        kernel_shape = np.array(symp_kernel_knots)[ii]
        return symp_kernel_amp * kernel_shape / kernel_shape.sum()

    @pm.potential
    def symp_days_to_infection_ratio(symp_kernel=symp_kernel):
        return pm.normal_like(symp_kernel.sum(), 5, .2**-2)

    @pm.deterministic(trace=True)
    def mu_symp(X=X, symp_kernel=symp_kernel):
        # add small offset to prevent zero-probability error
        return 1e-3 + my_convolve(X, symp_kernel)/data.n_pop * data.n_sampled

    @pm.observed
    def y_symp(mu=mu_symp, value=data.n_symp):
        return pm.poisson_like(value[10:], mu[10:])
    
    cc_kernel_amp = pm.Uniform('cc_kernel_amp', .01, .99, value=.1)
    cc_kernel_knots = [0, 0] + [pm.Uniform(f'cc_kernel_knot_{i}', 0, .1, value=.01) for i in range(5)]
    @pm.deterministic(trace=True)
    def cc_kernel(cc_kernel_amp=cc_kernel_amp, cc_kernel_knots=cc_kernel_knots):
        ii = np.linspace(0,7,num=21, endpoint=False)
        ii = np.array(np.floor(ii), dtype=int)
        kernel_shape = np.array(cc_kernel_knots)[ii]
        return cc_kernel_amp * kernel_shape / kernel_shape.sum()

    @pm.potential
    def cc_to_infection_ratio(cc_kernel=cc_kernel):
        return pm.normal_like(cc_kernel.sum(), 0.1, .02**-2)
    
    @pm.deterministic(trace=True)
    def mu_cc(X=X, cc_kernel=cc_kernel, beta_dow_cc=beta_dow_cc):
        cc_smooth = my_convolve(X, cc_kernel)
        return apply_day_of_week_effect(
            cc_smooth, beta_dow_cc, dow) + 1e-3 # add small offset to prevent zero-probability error
    @pm.observed
    def y_cc(mu=mu_cc, value=data.n_cc):
        return pm.poisson_like(value[10:], mu[10:])
    
    return locals()

def fit(var_dict, verbose=False):
    m = pm.MCMC(var_dict)
    m.use_step_method(pm.AdaptiveMetropolis, m.symp_kernel_knots[1:])
    m.use_step_method(pm.AdaptiveMetropolis, m.cc_kernel_knots[2:])
    for i in range(0, len(m.X_smooth)-7, 3):
        m.use_step_method(pm.AdaptiveMetropolis, m.X_smooth[i:(i+7)])

    for i in range(5):
        pm.MAP([m.y_cc, m.y_symp, m.X_smooth,
                m.X_smooth_potential]).fit(method='fmin_powell', verbose=verbose)
        pm.MAP([m.y_cc, m.y_symp, m.cc_kernel_knots
               ]).fit(method='fmin_powell', verbose=verbose)
        pm.MAP([m.y_cc, m.y_symp, m.symp_kernel_knots
               ]).fit(method='fmin_powell', verbose=verbose)
        pm.MAP([m.y_cc, m.y_symp, m.X_smooth,
                m.X_smooth_potential, m.beta_dow_cc_nonzero]).fit(method='fmin_powell', verbose=1)

    pm.MAP(var_dict).fit(method='fmin_powell', verbose=1)

    m.sample(50_000, 20_000, 30)

    return m

def my_plot(stoch, xx):
    s_mean = stoch.trace().mean(axis=0)
    s_lb, s_ub = pm.utils.hpd(stoch.trace(), 0.05)
    plt.plot(xx, s_mean, '-', color='C0', label='Predicted', zorder=3)
    plt.plot(xx, s_lb, '--', color='C0', zorder=3)
    plt.plot(xx, s_ub, '--', color='C0', zorder=3)

